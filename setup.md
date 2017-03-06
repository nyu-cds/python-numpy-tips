---
layout: page
title: Setup
permalink: /setup/
---
To complete this lesson, you will first need to install the `line_profiler` package in Anaconda. To do this, run the following
command from your shell (cmd on Windows):

~~~
$ conda install line_profiler
~~~
{: .shell}

Here is an example of the output yoiu should see. Enter `y` when asked to proceed.

~~~
Fetching package metadata .......
Solving package specifications: ..........

Package plan for installation in environment /Users/greg/anaconda3:

The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    conda-env-2.6.0            |                0          601 B
    line_profiler-2.0          |           py35_0          48 KB
    conda-4.3.13               |           py35_0         505 KB
    ------------------------------------------------------------
                                           Total:         554 KB

The following NEW packages will be INSTALLED:

    line_profiler: 2.0-py35_0               

The following packages will be UPDATED:

    conda:         4.2.13-py35_0 conda-forge --> 4.3.13-py35_0

The following packages will be SUPERCEDED by a higher-priority channel:

    conda-env:     2.6.0-0       conda-forge --> 2.6.0-0      

Proceed ([y]/n)? y

Pruning fetched packages from the cache ...
Fetching packages ...
conda-env-2.6. 100% |################################| Time: 0:00:00 480.42 kB/s
line_profiler- 100% |################################| Time: 0:00:00   1.20 MB/s
conda-4.3.13-p 100% |################################| Time: 0:00:00   5.31 MB/s
Extracting packages ...
[      COMPLETE      ]|###################################################| 100%
Unlinking packages ...
[      COMPLETE      ]|###################################################| 100%
Linking packages ...
[      COMPLETE      ]|###################################################| 100%
$ 
~~~
{: .output}
