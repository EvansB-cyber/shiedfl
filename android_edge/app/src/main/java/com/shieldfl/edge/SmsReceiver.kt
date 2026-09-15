package com.shieldfl.edge

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Intercepts incoming SMS at the OS level.
 * Lightweight edge role: extract text, hand off to local scorer.
 * Raw SMS text NEVER leaves this device — only engineered features + a
 * label (once the user confirms/denies) are ever transmitted to Provider.
 */
class SmsReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)
        val fullBody = messages.joinToString(separator = "") { it.messageBody ?: "" }
        val sender = messages.firstOrNull()?.originatingAddress ?: "unknown"

        Log.d("SmsReceiver", "Intercepted SMS from $sender, length=${fullBody.length}")

        // Process off the main thread — inference should never block the UI
        CoroutineScope(Dispatchers.Default).launch {
            EdgeScoringPipeline.getInstance(context).processIncomingSms(sender, fullBody)
        }
    }
}
