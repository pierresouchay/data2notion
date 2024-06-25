# Changelog for data2notion

## 1.0.3 - 2023-06-25

FEAT:

 - implemented --add-policy, --update-policy and --delete-policy to allow dry runs or
   confirmations
 - prometheus: better error handling and more precise error messages when queries are wrong
 - added --version to show current version information

## 1.0.2 - 2023-06-25

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
