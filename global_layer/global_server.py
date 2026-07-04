import torch
import torch.nn as nn
from edge_layer.models import SMSFraudCNN, CallDetectionMLP
from edge_layer.data import generate_global_test_data
from global_layer.server_optimizers import build_optimizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.crypto import encrypt_weights, decrypt_weights

class GlobalServer:
    """
    Tier 1: Central Global Server.
    Maintains the global master model, aggregates provider weights, and evaluates model validation performance.
    Supports pluggable server-side aggregation strategies (FedAvg / FedAdam / FedYogi / FedAdagrad).
    """
    def __init__(self, fl_algorithm="fedavg"):
        # Global models
        self.sms_model = SMSFraudCNN()
        self.call_model = CallDetectionMLP()
        
        # Validation test sets
        self.test_sms_dataset, self.test_call_dataset = generate_global_test_data(num_samples=150)
        
        # In-memory metrics logs
        self.metrics_history = []

        # FedOpt server-side adaptive optimization state (default = plain FedAvg,
        # which preserves the original aggregation behavior exactly)
        self.server_strategy_name = "fedavg"
        self.sms_server_optimizer = build_optimizer("fedavg")
        self.call_server_optimizer = build_optimizer("fedavg")

        if fl_algorithm == "fedopt":
            self.set_server_strategy("fedadam")
        
        # Evaluate initial random state
        self.log_metrics(round_id=0)

    def set_server_strategy(self, strategy_name, server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3):
        """
        Switches the server-side aggregation strategy. Raises ValueError if the
        strategy name is not recognized. Resets momentum/variance buffers since
        they are not meaningful across different strategies or hyperparameters.
        """
        self.server_strategy_name = strategy_name.lower()
        self.sms_server_optimizer = build_optimizer(strategy_name, server_lr, beta1, beta2, tau)
        self.call_server_optimizer = build_optimizer(strategy_name, server_lr, beta1, beta2, tau)

    def get_global_weights(self):
        return encrypt_weights(self.sms_model.state_dict()), encrypt_weights(self.call_model.state_dict())

    def set_global_weights(self, sms_state_dict, call_state_dict):
        self.sms_model.load_state_dict(sms_state_dict)
        self.call_model.load_state_dict(call_state_dict)

    def aggregate_provider_updates(self, provider_results_list, round_id=None):
        """
        Aggregates provider-level updates into the global model in two steps:
          1. Plain average across providers (standard FedAvg at the hierarchical mid-tier).
          2. Apply the configured server-side optimizer (FedAvg/FedAdam/FedYogi/FedAdagrad),
             treating the averaged update as a pseudo-gradient relative to the current
             global model (Reddi et al., ICLR 2021).
        With the default "fedavg" strategy, step 2 is a no-op and behavior is
        identical to the original implementation.
        """
        if not provider_results_list:
            return

        sms_weights_list = [decrypt_weights(res["sms_weights"]) for res in provider_results_list if res["sms_weights"] is not None]
        call_weights_list = [decrypt_weights(res["call_weights"]) for res in provider_results_list if res["call_weights"] is not None]

        if sms_weights_list:
            averaged_sms_weights = self._avg_weights(sms_weights_list)
            current_sms_weights = self.sms_model.state_dict()
            new_sms_weights = self.sms_server_optimizer.step(current_sms_weights, averaged_sms_weights)
            self.sms_model.load_state_dict(new_sms_weights)

        if call_weights_list:
            averaged_call_weights = self._avg_weights(call_weights_list)
            current_call_weights = self.call_model.state_dict()
            new_call_weights = self.call_server_optimizer.step(current_call_weights, averaged_call_weights)
            self.call_model.load_state_dict(new_call_weights)

    def get_global_state_dicts(self):
        """
        Returns the raw (unencrypted) global state dicts for SMS and call models.
        Kept for backward compatibility with CLI simulation.
        """
        return self.sms_model.state_dict(), self.call_model.state_dict()

    def _avg_weights(self, weights_list):
        avg_weights = {}
        for key in weights_list[0].keys():
            if weights_list[0][key].dtype.is_floating_point:
                tensors = [w[key] for w in weights_list]
                avg_weights[key] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                avg_weights[key] = weights_list[0][key].clone()
        return avg_weights

    def evaluate_model(self, model_type="sms"):
        """
        Evaluates accuracy on the global verification test dataset.
        """
        if model_type == "sms":
            model = self.sms_model
            dataset = self.test_sms_dataset
        else:
            model = self.call_model
            dataset = self.test_call_dataset
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataset:
                # Add batch dimension
                x_input = x.unsqueeze(0)
                outputs = model(x_input)
                _, predicted = torch.max(outputs, 1)
                total += 1
                if predicted.item() == y:
                    correct += 1
                    
        return correct / max(1, total)

    def _dataset_length(self, dataset):
        try:
            return len(dataset)
        except Exception:
            return sum(1 for _ in dataset)

    def evaluate_holdout(self):
        """
        Returns holdout evaluation summary used by CLI and tests.
        """
        sms_acc = self.evaluate_model("sms")
        call_acc = self.evaluate_model("call")
        sms_n = self._dataset_length(self.test_sms_dataset)
        call_n = self._dataset_length(self.test_call_dataset)
        return {
            "holdout_size": sms_n + call_n,
            "holdout_sms_accuracy": round(sms_acc, 4),
            "holdout_call_accuracy": round(call_acc, 4),
        }

    def log_metrics(self, round_id):
        """
        Calculates and logs verification metrics for the current round.
        """
        sms_acc = self.evaluate_model("sms")
        call_acc = self.evaluate_model("call")

        sms_n = self._dataset_length(self.test_sms_dataset)
        call_n = self._dataset_length(self.test_call_dataset)
        holdout_size = sms_n + call_n

        metrics = {
            "round": round_id,
            "sms_accuracy": round(sms_acc, 4),
            "call_accuracy": round(call_acc, 4),
            "holdout_sms_accuracy": round(sms_acc, 4),
            "holdout_call_accuracy": round(call_acc, 4),
            "holdout_size": holdout_size,
            "server_strategy": self.server_strategy_name
        }
        self.metrics_history.append(metrics)
        return metrics