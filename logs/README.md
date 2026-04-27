<!-- @summary
Runtime log output directory. Log files are auto-generated at runtime and are not committed to version control.
@end-summary -->

# logs/

This directory holds log files written at runtime by the RAG stack processes. Files here are transient — they are created when a process starts and are not tracked in version control.

## Log files

Log files are named after the process that writes them (e.g. `rag_query.log`, `query_processor.log`, `cli_client.log`). They are created on first run and grow until rotated or cleared. Do not commit them — add any new log filenames to `.gitignore` if they are not already covered.
