"""List of Models in stan"""

class Modeller:
    def __init__(self):
        pass

    @staticmethod
    def blr():
        """
        Model for Bayesian linear regression
        """


        blr = """
        data {
            int<lower=1> N; //number of samples
            vector[N] x; // samples
            vector[N] y; // responses
        }
        parameters {
            real a; // intercept
            real b; // slope
        }
        model {
            a~normal(0,10);
            b~normal(0,10);
            y~normal(a + b*x, 0.5); // main model
        }
        """
        return blr

    @staticmethod
    def probit_reg():
        """Model for probit regression"""
        probit = """
        data {
            int<lower=1> N; //number of samples
            vector[N] x; // samples
            int<lower=0, upper=1> y[N]; // responses
        }
        parameters {
            real a; // intercept
            real b; // slope
        }
        transformed parameters {
            vector[N] mu;
            mu = Phi(a+b*x);
        }
        model {
            a~normal(0,10);
            b~normal(0,10);
            y ~ bernoulli(mu);
              
        }
        """
        return probit
