# Changelog for data2notion

## Upcoming

## 1.0.12 - 2025-04-09

FIX:

- fix buggy plugin filtering that cause import/export plugins to fail

## 1.0.11 - 2025-04-09

FEAT:

- [jagu-sayan](https://github.com/jagu-sayan) added feature to export Notion into CSV
  This feature supports CSV plugin only for now and can be used with `export-from-notion` command.

## 1.0.10 - 2025-04-07

PERF:

- Propery compare timestamps of source/notion when source is more precise
  than notion (which has minute-level granularity) -> fewer diffs -> more performance  

## 1.0.9 - 2025-03-24

FEAT:

 - Support for partitioning synchronization
   using `--partition tags=(tag1|tag2)` will now only sync elements from
   source and destination when their property named 'tags' contains either tag1 or tag2.
   This is helpful to partially synchronize several CSV files into the same database for instance.
   This supports regexp, so possible to use `--partition 'tags=prefix1.*'` for instance

## 1.0.8 - 2025-03-18

FIX:

 - bump asynciolimiter ≥ 1.1.2 to fix possible deadlock when performing many updates

## 1.0.7 - 2025-03-10

FEAT:

 - improve progressbar during updates

FIX:

 - improve possible deadlocks for databases with huge number of entries to update
 - dependency for notion_client to 2.x version

## 1.0.6 - 2025-03-06

PERF:

 - added `--notion-rate-limiter` option to specify rate limit

FEAT:

 - added progressbar (can be deactivated with --no-progress-bar)

## 1.0.5 - 2024-10-16

PERF:

 - Properly diff multi-select to avoid always trying to update values

## 1.0.4 - 2024-06-26

FEAT:

 - Detect Too many requests and shutdown immediatly
 - Display URL, title and information at startup when syncing
 - Improved display of statistics

## 1.0.3 - 2024-06-25

FEAT:

 - implemented --add-policy, --update-policy and --delete-policy to allow dry runs or
   confirmations
 - prometheus: better error handling and more precise error messages when queries are wrong
 - added --version to show current version information

## 1.0.2 - 2024-06-25

FIX:

 - Improve error messages in Prometheus when evaluating expression / be sure to have
   all labels available in --row-id-expression

## 1.0.1 - 2024-06-25

Minor version: doc and statistics

FEATURES:

 - added statistics to track performance `--statistics console` to see it in action

DOCUMENTATION:

 - added documentation to write plugins
 - improved README.md and DESIGN.md
 - Added CHANGELOG.md

## 1.0.0 - 2024-06-24

FEATURES:

 - plugin architecture
 - 3 plugins: csv, json and prometheus
 - Async code for good performance
