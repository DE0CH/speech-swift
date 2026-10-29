import MLX
import MLXCommon

/// MLX's memory figures and a cap on its buffer cache, for callers of Qwen3ASR that don't import MLX themselves.
public enum Qwen3ASRMemory {
    /// Bytes held by live arrays (weights included).
    public static var active: Int { Memory.activeMemory }
    /// Bytes of freed buffers MLX keeps for reuse.
    public static var cached: Int { Memory.cacheMemory }
    /// The most `active` has been since the process started.
    public static var peak: Int { Memory.peakMemory }

    /// Keeps at most `bytes` of freed buffers for reuse while the returned token is held (the tightest of all held
    /// limits applies, so this lowers the default models set for themselves); the rest goes back to the system.
    public static func limitCache(to bytes: Int) -> AnyObject? {
        MLXCacheLimitCoordinator.shared.acquire(ceiling: bytes)
    }
}
