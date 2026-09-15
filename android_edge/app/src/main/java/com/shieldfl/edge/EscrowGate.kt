package com.shieldfl.edge

import android.content.Context
import android.util.Log

/**
 * Pure local decision logic — no network call needed to hold a transaction.
 * This is what protects the user in real time, independent of whether
 * the device is online or has synced with the Provider recently.
 */
object EscrowGate {

    fun holdIfTransactionFollows(context: Context, record: PendingSmsRecord) {
        // In the full prototype, this hooks into whatever intercepts the
        // transaction-approval flow (e.g. a notification listener watching
        // for MoMo USSD/app approval prompts within a short window after
        // a high-risk SMS). For the prototype, this raises a local flag
        // that the transaction-approval UI checks before letting an
        // OTP/PIN entry go through.
        Log.w("EscrowGate", "HOLD triggered for SMS from ${record.sender}, risk=${record.riskScore}")

        EdgeDatabase.getInstance(context).escrowFlagDao().raiseFlag(
            EscrowFlag(
                relatedSmsId = record.id,
                reason = "smishing_risk",
                riskScore = record.riskScore,
                active = true,
                timestamp = System.currentTimeMillis()
            )
        )

        // TODO: trigger a user-facing notification/dialog here:
        // "A recent message looks suspicious. A transaction approval has
        //  been held. Review before confirming with your PIN."
    }
}
