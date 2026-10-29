import MLX
import MLXCommon

/// MLX's memory figures and a cap on its buffer cache, for callers of Qwen3ASR that don't import MLX themselves.
public enum Qwen3ASRMLXMemory {
    /// Bytes held by live arrays (weights included).
    public static var active: Int { MLX.Memory.activeMemory }
    /// Bytes of freed buffers MLX keeps for reuse.
    public static var cached: Int { MLX.Memory.cacheMemory }
    /// The most `active` has been since the process started.
    public static var peak: Int { MLX.Memory.peakMemory }

    /// Keeps at most `bytes` of freed buffers for reuse while the returned token is held (the tightest of all held
    /// limits applies, so this lowers the default models set for themselves, e.g. 1.7B's min(4 GB, a quarter of RAM)); the rest
    /// goes back to the system.
    public static func limitCache(to bytes: Int) -> AnyObject? {
        MLXCacheLimitCoordinator.shared.acquire(ceiling: bytes)
    }
}
