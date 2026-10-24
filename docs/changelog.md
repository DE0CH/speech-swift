# Changelog

## Unreleased

- Preserve the final attempt's `DownloadError.stalled`, `checksumMismatch`,
  `invalidRemoteFileName`, and `networkUnavailable` cases and associated values
  after HuggingFace download retries. A stall previously became
  `networkUnavailable`; callers handling connectivity failures should now handle
  `stalled` too. Complete-cache fallback for stalls and other connectivity
  failures is unchanged. HTTP failures still use `failedToDownload` with retry
  context. See [download errors](shared-protocols.md#downloaderror).
