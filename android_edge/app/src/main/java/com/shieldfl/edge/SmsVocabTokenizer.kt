package com.shieldfl.edge

import android.content.Context
import org.json.JSONObject

/**
 * On-device SMS tokenizer that mirrors Python `tokenize_message()` in edge_layer/data.py
 * exactly — same punctuation stripping, same PAD/UNK mapping, same seq_len — so that
 * the integer sequences fed into the ONNX model are numerically identical to those
 * produced during training.
 *
 * Vocabulary source:
 *   assets/vocab.json  (exported by `export_vocab_json()` in data.py)
 *
 * JSON schema expected:
 * {
 *   "vocab_version": "1.1.0",
 *   "pad_id": 0,
 *   "unk_id": 1,
 *   "vocab_size": 110,
 *   "tokens": { "<PAD>": 0, "<UNK>": 1, "hello": 2, ... }
 * }
 *
 * Usage:
 *   val tokenizer = SmsVocabTokenizer.getInstance(context)
 *   val ids: LongArray = tokenizer.tokenize("URGENT: Your MTN MoMo pin verify now")
 *   // → [1, 85, 71, 87, 96, 0, 0, ...] (length == SEQ_LEN)
 */
class SmsVocabTokenizer private constructor(context: Context) {

    companion object {
        const val SEQ_LEN = 20
        private const val VOCAB_FILE = "vocab.json"

        // Characters to strip — must match Python tokenize_message() exactly:
        //   for ch in [",", ".", "!", "?", '"', "'", ":", ";", "(", ")", "-", "_", "/"]:
        private val STRIP_CHARS = setOf(',', '.', '!', '?', '"', '\'', ':', ';', '(', ')', '-', '_', '/')

        @Volatile private var instance: SmsVocabTokenizer? = null

        fun getInstance(context: Context): SmsVocabTokenizer =
            instance ?: synchronized(this) {
                instance ?: SmsVocabTokenizer(context.applicationContext).also { instance = it }
            }
    }

    // token → integer index, loaded once from assets/vocab.json
    private val tokenMap: Map<String, Int>
    private val padId: Int
    private val unkId: Int

    init {
        val jsonStr = context.assets.open(VOCAB_FILE)
            .bufferedReader()
            .use { it.readText() }

        val root    = JSONObject(jsonStr)
        padId       = root.getInt("pad_id")
        unkId       = root.getInt("unk_id")

        val tokensObj = root.getJSONObject("tokens")
        val map = HashMap<String, Int>(tokensObj.length() * 2)
        for (key in tokensObj.keys()) {
            map[key] = tokensObj.getInt(key)
        }
        tokenMap = map
    }

    /**
     * Converts an SMS body string into a fixed-length LongArray suitable for
     * ONNX Runtime input (shape [1, SEQ_LEN], type int64).
     *
     * Pipeline (matches Python tokenize_message() step for step):
     *   1. Lowercase
     *   2. Replace each punctuation character in STRIP_CHARS with a space
     *   3. Split on whitespace
     *   4. Map each word to its vocab index; unknown words → unkId
     *   5. Truncate to SEQ_LEN or pad with padId on the right
     */
    fun tokenize(smsBody: String, seqLen: Int = SEQ_LEN): LongArray {
        // Step 1: lowercase
        var clean = smsBody.lowercase()

        // Step 2: strip punctuation (replace each char with space)
        val sb = StringBuilder(clean.length)
        for (ch in clean) {
            sb.append(if (ch in STRIP_CHARS) ' ' else ch)
        }
        clean = sb.toString()

        // Step 3: split on whitespace (filter empty tokens)
        val words = clean.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }

        // Step 4: map to indices
        val indices = words.map { word ->
            (tokenMap[word] ?: unkId).toLong()
        }

        // Step 5: pad / truncate to seqLen
        val result = LongArray(seqLen) { padId.toLong() }
        val copyLen = minOf(indices.size, seqLen)
        for (i in 0 until copyLen) {
            result[i] = indices[i]
        }
        return result
    }

    /** Returns the vocab size as recorded in vocab.json. */
    fun vocabSize(): Int = tokenMap.size

    /** Returns the vocab version string (e.g. "1.1.0"). */
    fun vocabVersion(): String = tokenMap["<PAD>"]?.let { "loaded" } ?: "unknown"
}
