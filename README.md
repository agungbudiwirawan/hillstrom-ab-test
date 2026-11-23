# Hillstrom email A/B test

Does a promotional email lead to more purchases, and if you can only send one version, which one should it be?
This project answers both questions with a randomised experiment on 64,000 retail customers.

The full analysis is in [`notebooks/hillstrom_ab_test.ipynb`](notebooks/hillstrom_ab_test.ipynb).

## The experiment

In 2008 Kevin Hillstrom (MineThatData) published data from an email test run by a retailer that sells men's and
women's merchandise. Customers who had bought something in the past year were split at random into three groups:

| Group | What they received | Customers |
|---|---|---|
| Mens | Email featuring men's merchandise | 21,307 |
| Womens | Email featuring women's merchandise | 21,387 |
| Control | No email | 21,306 |

For two weeks after the send, the retailer recorded visits, purchases and spend for every customer.

## Results

Purchase conversion is the primary metric. Visits and spend are secondary.

| Metric | Control | Womens | Mens |
|---|---|---|---|
| Conversion rate | 0.57% | 0.88% | 1.25% |
| Visit rate | 10.6% | 15.1% | 18.3% |
| Spend per customer | $0.65 | $1.08 | $1.42 |

The Mens email roughly doubles conversion compared with sending nothing: +0.68 percentage points, 95% CI 0.50 to 0.86.
It also beats the Womens email by 0.37 points (CI 0.17 to 0.56). Every pairwise difference in conversion and visits
stays significant after a Holm correction for three comparisons.

On spend, the Mens email adds about $770 of revenue per 1,000 emails over two weeks (CI $485 to $1,055).
The Womens email adds about $424.

A Bayesian model of the conversion rates gives the Mens email a 99.99% probability of being the best of the three.

**Recommendation:** send the Mens email. If the team can target by segment, the next experiment worth running is the
Womens email for customers who bought women's items last year and the Mens email for everyone else.
The data points that way, but the effect was not significant after correction, so it needs its own test.

## How the analysis works

1. Sanity checks. A chi-square test for sample ratio mismatch (p = 0.90) and standardised mean differences on
   every pre-experiment variable (all below 0.02) confirm the randomisation worked.
2. Power check. With about 21,300 customers per group the test can reliably detect a relative lift of roughly 46%
   in conversion, so small effects could have been missed.
3. Conversion and visits. An omnibus chi-square test on the 3 x 2 table, then pairwise two-proportion z-tests with
   confidence intervals and Holm-adjusted p-values.
4. Spend. Welch's t-test, confirmed with a bootstrap confidence interval because 99% of customers spent $0.
   Mann-Whitney U is reported as a robustness check.
5. Regression adjustment. Adding customer covariates changes the standard errors by less than 1%, so the simple
   comparisons are already as precise as this data allows.
6. Segments. Logistic regressions with group-by-segment interactions, tested with likelihood ratio tests.
   The email effect differs by customer tenure (Holm p = 0.02). New customers barely buy without an email,
   and both emails lift them a lot.
7. Bayesian view. Beta-Binomial posteriors for each group's conversion rate, with the probability of being best
   and the expected loss of each choice.

## Limitations

The outcomes cover only two weeks. There is no data on unsubscribes, margins or send costs, so the revenue figures
are not profit. The segment analysis is exploratory. The test ran in 2008 on one retailer's customers, so the
direction of the result is more likely to carry over than the exact sizes.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/hillstrom_ab_test.ipynb
```

The notebook downloads the data into `data/` on first run. The CSV is not committed to this repository.
`build_notebook.py` regenerates the notebook from source if you want to edit the cells as Python.

## Data source

Kevin Hillstrom, "The MineThatData E-Mail Analytics And Data Mining Challenge", MineThatData blog, March 2008.
All credit for the data goes to the author.
