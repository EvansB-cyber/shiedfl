package com.shieldfl.edge

import android.content.Context
import androidx.room.*

@Entity(tableName = "pending_sms")
data class PendingSmsRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sender: String,
    val featuresJson: String,   // engineered features only — never raw SMS text
    val riskScore: Float,
    val timestamp: Long,
    val userLabel: Int? = null, // null = no feedback yet, 1 = confirmed fraud, 0 = false positive
    val synced: Boolean = false
)

@Entity(tableName = "escrow_flags")
data class EscrowFlag(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val relatedSmsId: Long,
    val reason: String,
    val riskScore: Float,
    val active: Boolean,
    val timestamp: Long
)

@Dao
interface PendingSmsDao {
    @Insert
    fun insert(record: PendingSmsRecord): Long

    @Query("SELECT * FROM pending_sms WHERE synced = 0 AND userLabel IS NOT NULL")
    fun getUnsyncedLabeled(): List<PendingSmsRecord>

    @Query("UPDATE pending_sms SET userLabel = :label WHERE id = :id")
    fun setLabel(id: Long, label: Int)

    @Query("UPDATE pending_sms SET synced = 1 WHERE id IN (:ids)")
    fun markSynced(ids: List<Long>)

    @Query("SELECT * FROM pending_sms ORDER BY timestamp DESC LIMIT 50")
    fun getRecent(): List<PendingSmsRecord>
}

@Dao
interface EscrowFlagDao {
    @Insert
    fun raiseFlag(flag: EscrowFlag): Long

    @Query("SELECT * FROM escrow_flags WHERE active = 1 ORDER BY timestamp DESC")
    fun getActiveFlags(): List<EscrowFlag>

    @Query("UPDATE escrow_flags SET active = 0 WHERE id = :id")
    fun clearFlag(id: Long)
}

@Database(entities = [PendingSmsRecord::class, EscrowFlag::class], version = 1)
abstract class EdgeDatabase : RoomDatabase() {
    abstract fun pendingSmsDao(): PendingSmsDao
    abstract fun escrowFlagDao(): EscrowFlagDao

    companion object {
        @Volatile private var instance: EdgeDatabase? = null

        fun getInstance(context: Context): EdgeDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    EdgeDatabase::class.java,
                    "shieldfl_edge.db"
                ).build().also { instance = it }
            }
    }
}
