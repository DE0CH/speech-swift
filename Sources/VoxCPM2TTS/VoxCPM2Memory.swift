import Foundation
import MLXCommon

enum VoxCPM2Memory {
    /// Bound MLX's reusable-buffer cache while the model is loaded. Each line has a different length, so a process
    /// that keeps the model and never clears the cache collects buffers of new sizes: about 29 GB after ten lines on a
    /// 48 GB Mac, on top of the 9.3 GB of float32 weights.
    static func cacheLimit(physicalMemoryBytes: Int) -> Int {
        max(0, min(2 * 1024 * 1024 * 1024, physicalMemoryBytes / 4))
    }

    static func acquireCacheLimit(
        physicalMemoryBytes: Int = Int(ProcessInfo.processInfo.physicalMemory),
        coordinator: MLXCacheLimitCoordinator = .shared
    ) -> MLXCacheLimitCoordinator.Lease? {
        coordinator.acquire(ceiling: cacheLimit(physicalMemoryBytes: physicalMemoryBytes))
    }
}
