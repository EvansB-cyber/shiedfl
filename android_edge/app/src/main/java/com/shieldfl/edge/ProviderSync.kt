package com.shieldfl.edge

import android.content.Context
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST
import retrofit2.Call

data class FeedbackPayload(
    val featuresJson: String,
    val label: Int,
    val riskScore: Float,
    val deviceId: String
)

data class FeedbackResponse(val status: String)

interface ProviderApi {
    // Matches your existing FastAPI route pattern, e.g. POST /edge/feedback
    @POST("edge/feedback")
    fun submitFeedback(@Body payload: FeedbackPayload): Call<FeedbackResponse>
}

object ProviderSync {

    // Point this at your Provider tier — swap for your Render URL or local
    // provider endpoint. Never point this at Global directly; Provider
    // absorbs the raw feedback and does the heavy training.
    private const val BASE_URL = "https://shieldfl.onrender.com/"

    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    private val api = retrofit.create(ProviderApi::class.java)

    fun syncPendingFeedback(context: Context, deviceId: String) {
        val dao = EdgeDatabase.getInstance(context).pendingSmsDao()
        val unsynced = dao.getUnsyncedLabeled()
        if (unsynced.isEmpty()) return

        val syncedIds = mutableListOf<Long>()
        for (record in unsynced) {
            val payload = FeedbackPayload(
                featuresJson = record.featuresJson,
                label = record.userLabel ?: continue,
                riskScore = record.riskScore,
                deviceId = deviceId
            )
            try {
                val response = api.submitFeedback(payload).execute()
                if (response.isSuccessful) {
                    syncedIds.add(record.id)
                }
            } catch (e: Exception) {
                // Network unavailable — leave unsynced, retry on next trigger
            }
        }
        if (syncedIds.isNotEmpty()) {
            dao.markSynced(syncedIds)
        }
    }
}
