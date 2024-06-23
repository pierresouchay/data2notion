# data2notion: export your data to Notion

This tool is intended for exporting various data from 3rd party systems
into Notion. If you tool can export some records, then you can make this data available to Notion!

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

