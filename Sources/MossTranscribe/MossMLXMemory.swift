import Foundation
import MLXCommon

enum MossMLXMemory {
    /// Bound reusable buffers independently of the live long-context KV cache.
    /// Clearing buffers between encoder batches does not bound decoder caching.
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
