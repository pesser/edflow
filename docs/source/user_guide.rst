User Guide
==========

Old text:
Writing training loops and dataset classes can be fun but usually gets
repetitive and a waste of time after having done it a few times.

But there is a solution: ``edflow`` does all the dirty work for you and helps
you make reusable models, datasets and trainers. This way you can evaluate
reliably and fast, which makes you work more productive as you can track and
compare experiments.

All this without restricting you too much or making assumptions about the 
deep learning framework you use.

So what are you waiting for? Use ``edflow``!

Now, we know that all might be not as perfect as it sounds above, but we think
we are on the right track to a good training and evaluation framework for
deep learning models. Below we take you through the core concepts and explain
the our ideas behind them. You will learn about

* The overall workflow
* Datasets
* Iterators
* Hooks
* Models
* Evaluation

If you want to dig in deeper take a look at our advanced guide, which tells
you more about
* logging
* the project manager
* error handling
