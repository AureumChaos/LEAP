# LEAP
A general purpose Library for Evolutionary Algorithms in Python.
Written by Dr. Jeffrey K. Bassett
Contributors: Dr. Mark Coletti, Eric Scott, Dr. R. Paul Weigand

Much of the library design was inspired by the general description of
Evolutionary Algorithms from Dr. Kenneth A. De Jong's book "Evolutionary
Computation: A Unified Approach".  He was my Ph.D. advisor, so I am quite
familiar with his approach.

As of 10/6/2018:

I wouldn't exactly call this stable just yet, but anyone is free to use it if
they wish.  The current license is GPL v2, but I'm likely to change that to
something that is more permissive and more academic (the MIT license?) at some
point in the future.

I wrote this library while I was working on my dissertation some years back.
I had always thought to release it, but never actually did.  As I look over
the code now, I see that there are some pieces that I'm not completely happy
with.  It's clear that I was learning Python at the same time that I wrote
much of this.

I've now upgraded the entire library to Python 3.  Everything at the top level
seems to work fine in both Python 2 and 3.  I've tried to upgrade the
sub-modules as well, but there are a few problems caused by changes in the way
modules work in Python 3.  Unfortunately I haven't been able to find a simple
fix for these.  I may need to reorganize the code in order to get it to work.
From the looks of it, everything still works in Python 2 though.

