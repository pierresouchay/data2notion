# Design of data2notion

## Overview

data2notion synchronizes a source (CSV, database) to a Notion database (the sink).

The data from the source is provided using plugins, the plugins will
act as a source of truth and will add/delete/modify data in the Notion
Sink.

Plugins can be written to use Notion as a frontend to the source, very helpfull
to comment, annotate data from various systems.

## Plugins

Plugins provide ways to access the data from the so
