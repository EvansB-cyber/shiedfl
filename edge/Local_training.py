# edge/local_training.py
import torch
from torch.optim import SGD

LOCAL_LR = 0.01
LOCAL_EPOCHS = 3

def local_update(model: SMSFraudCNN, labeled_batch: list) -> dict:
    optimizer = SGD(model.parameters(), lr=LOCAL_LR)
    model.train()
    for epoch in range(LOCAL_EPOCHS):
        for sms_tokens, label in labeled_batch:
            optimizer.zero_grad()
            output = model(sms_tokens)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output, torch.tensor([float(label)])
            )
            loss.backward()
            optimizer.step()
    return model.state_dict()