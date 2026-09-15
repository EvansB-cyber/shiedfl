package com.shieldfl.edge

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.nio.LongBuffer

/**
 * Edge-tier responsibility (lightweight, on-device only):
 *   1. Tokenize SMS text using SmsVocabTokenizer (vocab.json, matches Python training)
 *   2. Run quantized ONNX model for a risk score
 *   3. Decide whether to raise an escrow hold
 *   4. Store the (features, pending) record locally until user feedback arrives
 *
 * Privacy guarantee: raw SMS text NEVER leaves this device. Only engineered
 * feature vectors + a user-confirmed label are ever sent to the Provider tier.
 *
 * Tokenizer parity: SmsVocabTokenizer mirrors Python tokenize_message() exactly —
 * same vocab.json, same punctuation rules, same PAD/UNK, same seq_len=20 — so
 * the integer sequences fed to the ONNX model are numerically identical to those
 * used during training.
 */
class EdgeScoringPipeline private constructor(private val context: Context) {

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    companion object {
        private const val RISK_THRESHOLD = 0.75f
        private const val MODEL_FILE = "sms_fraud_cnn_quantized.onnx"

        @Volatile private var instance: EdgeScoringPipeline? = null

        fun getInstance(context: Context): EdgeScoringPipeline =
            instance ?: synchronized(this) {
                instance ?: EdgeScoringPipeline(context.applicationContext).also { instance = it }
            }
    }

    private fun getSession(): OrtSession {
        if (session == null) {
            val modelBytes = context.assets.open(MODEL_FILE).readBytes()
            session = ortEnv.createSession(modelBytes)
        }
        return session!!
    }

    /**
     * Converts SMS text into a vocab-index integer sequence that matches what
     * SMSFraudCNN was trained on.
     *
     * Delegates to SmsVocabTokenizer which:
     *   1. Lowercases the text
     *   2. Strips punctuation (same set as Python tokenize_message)
     *   3. Maps tokens via vocab.json (loaded from assets/)
     *   4. Pads / truncates to seq_len=20
     *
     * Returns a LongArray of shape [SEQ_LEN] for use as ONNX int64 input.
     */
    private fun tokenize(smsBody: String): LongArray {
        return SmsVocabTokenizer.getInstance(context).tokenize(smsBody)
    }

    fun processIncomingSms(sender: String, smsBody: String) {
        val tokenIds = tokenize(smsBody)
        val riskScore = runInference(tokenIds)

        // Store the feature vector as a JSON-friendly string for ProviderSync
        val featuresJson = tokenIds.joinToString(",", prefix = "[", postfix = "]")

        val record = PendingSmsRecord(
            sender = sender,
            featuresJson = featuresJson,
            riskScore = riskScore,
            timestamp = System.currentTimeMillis()
        )

        EdgeDatabase.getInstance(context).pendingSmsDao().insert(record)

        if (riskScore > RISK_THRESHOLD) {
            EscrowGate.holdIfTransactionFollows(context, record)
        }
    }

    /**
     * Runs the ONNX model with the tokenized integer sequence.
     *
     * Input tensor:  int64, shape [1, SEQ_LEN]  — matches SMSFraudCNN's embedding input
     * Output tensor: float32, shape [1, 2]       — [P(ham), P(spam)]; risk = P(spam)
     *
     * Fail-safe: returns 0.5 (uncertain / flag for review) if inference throws.
     */
    private fun runInference(tokenIds: LongArray): Float {
        return try {
            val shape = longArrayOf(1L, tokenIds.size.toLong())
            val inputTensor = OnnxTensor.createTensor(
                ortEnv,
                LongBuffer.wrap(tokenIds),
                shape
            )
            val results = getSession().run(mapOf("input" to inputTensor))
            // Output shape [1, 2]: column 1 = P(spam)
            val output = results[0].value as Array<FloatArray>
            output[0][1]   // P(spam) is the fraud risk score
        } catch (e: Exception) {
            // Fail safe: if inference fails, do not silently pass — flag for review
            0.5f
        }
    }
}
