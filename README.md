# data2notion: export your data to Notion

[![PyPI - Version](https://img.shields.io/pypi/v/data2notion)](https://pypi.org/project/data2notion/)
[![License](https://img.shields.io/pypi/l/data2notion)](https://raw.githubusercontent.com/pierresouchay/data2notion/main/LICENSE)

This tool is intended for exporting various data from 3rd party systems
into Notion. If you tool can export some records, then you can make this data available to Notion!

## Installation

It requires Python 3.9+ and try not using too many dependencies.

to install it:

```bash
python3 install data2notion
```

This will install it and all plugins. You can now run it by typing `data2notion` in you terminal.

## Changes

[CHANGELOG.md](https://github.com/pierresouchay/data2notion/blob/main/CHANGELOG.md)

## Goals

When working as a CTO, I had the need to expose data from various systems (Applications, AWS, SIEM, Active Directory, users, groups...)
into Notion to generate automatic reports and select what I wanted to expose, so it made all systems transparent and open.
This was extremelly precious to automatically generate reports for audits.

One of the advantages of the tool is that:

 - it creates entries in Notion database if entries do not exists
 - it modifies the various fields of the Notion database if they have been modified
 - it cleanups the old records when they don't exist anymore

So, if you run this tool everyday, you have a Notion database in sync with the data from all your third party systems.
And you can start commenting, taking decisions and documenting those entries within Notion: it gives you a centralized
way to discuss topics about all of your IT, provides comments, reminders (eg: reming me about renewing this certificate
in 2 years, check if the fix has been performed in 2 months...), and cleanup dead stuff automatically.

Initially, the tool was only dealing with CSV, but data2notion also deals with various formats and APIs, so you can
integrate faster and better!

# Features

data2notion supports plugins, so you can either work with JSON/CSV export or implement your own retrieval of data, for instance,
from a 3rd party API.

## Basic Usage

data2notion is build using commands. Each succession of command has its own help.

The first level is teh following:

```bash
data2notion --help
usage: data2notion [-h] [--version] [--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}] [--notion-log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}]
                   [--notion-rate-limit NOTION_RATE_LIMIT] [--statistics {console,disabled}] [--notion-token NOTION_TOKEN]
                   {plugins,write-to-notion} ...

Export some data into a notion database

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        Set the default log level (default=WARNING)
  --notion-log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        Set the log level for data2notion (default=INFO)
  --notion-rate-limit NOTION_RATE_LIMIT
                        Set the notion rate-limiter, by default {default_rate_limit} (3 requests/sec, 100 initial bucket size)
  --statistics {console,disabled}
                        Display Statistics when program ends
  --notion-token NOTION_TOKEN
                        Notion Token to use $NOTION_TOKEN by default

action to perform:
  action to perform

  {plugins,write-to-notion}
                        sub-command help
    write-to-notion     write to Notion Database
```

This is the level where you can specify using the command line the `NOTION_TOKEN` (or you can use the environemnt variable `$NOTION_TOKEN` instead).

You can also change the log-level (in case of issues) and enable/disable statistics.

### First level of help

How to get the _notion_database_id_?

From [official documentation](https://developers.notion.com/reference/retrieve-a-database), open the database in your brower (in a full page),
the URL should then be https://notion.so/my_workspace/**668d797c76fa49349b05ad288df2d136**v=... => in such an example, `668d797c76fa49349b05ad288df2d136`
would be the _notion_database_id_.

Be sure that you [granted access rights to this database](https://www.notion.so/help/add-and-manage-connections-with-the-api#add-connections-to-pages)
to the token you want to use in this app.


```bash
data2notion write-to-notion <notion_database_id> <plugin> --help
```

Will list the arguments that are specific to a plugin.

### CSV plugin

Export CSV files arrays in Notion.

```bash
data2notion write-to-notion <notion_database_id> csv --help
usage: data2notion write-to-notion notion_database_id csv [-h] [--csv-dialect {excel,excel-tab,unix,excel_with_semi-colon}] csv_file

Write to Notion DB from a CSV file

positional arguments:
  csv_file              CSV file to inject in Notion, if '-' is set, read from stdin

options:
  -h, --help            show this help message and exit
  --csv-dialect {excel,excel-tab,unix,excel_with_semi-colon}
                        CSV Dialect, excel by default
```

### JSON Plugin

Export some JSON arrays in Notion.

```bash
usage: data2notion write-to-notion <notion_database_id> json [-h] [--json-path PATH_IN_JSON] json_file

Write to Notion DB from a JSON file containing an array

positional arguments:
  json_file             JSON file to inject in Notion, if '-' is set, read from stdin

options:
  -h, --help            show this help message and exit
  --json-path PATH_IN_JSON
                        JSON path separated by dots to look for the array, example: calendar.appointments
```

### Prometheus Plugin

Export metrics from prometheus (lastest values only) in Notion database.

Inspired by [prom2csv](https://pypi.org/project/prom2csv/), this plugin let you export your last prometheus metrics in Notion!

```bash
usage: data2notion [-h] [--version] [--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}] [--notion-log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}] [--no-progress-bar] [--notion-rate-limit NOTION_RATE_LIMIT] [--statistics {console,disabled}]
                   [--notion-token NOTION_TOKEN] [--partition column_name=<regexp>]
                   {plugins,export-from-notion,write-to-notion} ...

Export some data into a notion database

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        Set the default log level (default from $NOTION_LOG_LEVEL=WARNING)
  --notion-log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        Set the log level for data2notion (default=INFO)
  --no-progress-bar     Disable the progress bar
  --notion-rate-limit NOTION_RATE_LIMIT
                        Set the notion rate-limiter, by default {default_rate_limit} (3 requests/sec, 100 initial bucket size)
  --statistics {console,disabled}
                        Display Statistics when program ends
  --notion-token NOTION_TOKEN
                        Notion Token to use $NOTION_TOKEN by default
  --partition column_name=<regexp>
                        Only synchronize records having a column matching given regexp

action to perform:
  action to perform

  {plugins,export-from-notion,write-to-notion}
                        sub-command help
    export-from-notion  export from Notion Database
    write-to-notion     write to Notion Database
```


## Adding other plugins


Adding plugin can be done using environment variable `$DATA2NOTION_ADDITIONAL_PLUGINS` which support the
following syntax `python_module.python_file:PluginClass`, thus to enable a new imaginary Bugzilla plugin, you might for instance
export the variable:

```bash
export DATA2NOTION_ADDITIONAL_PLUGINS="bugzilla_api.bugzilla_plugin:BugzillPlugin"
```

Then, the plugin should be visible in:
```bash
data2notion plugins
```

and you might use it as any other plugin.

## Writing new plugins

See [DESIGN.md](https://github.com/pierresouchay/data2notion/blob/main/DESIGN.md).
