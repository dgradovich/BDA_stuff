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

    @staticmethod
    def logistic_reg():
        """Logistic regression model with both parameters"""
        log_reg = """

            data {
            int<lower=0> N;
            real x[N];
            int<lower=0> y[N];
            }

            parameters {
            real alpha;
            real beta;
            }

            transformed parameters {
            real p[N];
            for (i in 1:N){
                    p[i] = inv_logit(beta*x[i] + alpha);
                    }
            }
            model {
            alpha~normal(0,10);
            beta~normal(0,10);
            y~binomial(5,p);
            }
            generated quantities {
            real ld50; // ld50 parameter
            ld50 = -alpha/beta;
            }
        """
        return log_reg

    @staticmethod
    def logistic_reg_fixed_beta():
        """Logistic regression model with beta fixed to 10"""
        log_reg = """

            data {
            int<lower=0> N;
            real x[N];
            int<lower=0> y[N];
            real beta; //fixed beta parameter
            }

            parameters {
            real alpha;
            }

            transformed parameters {
            real p[N];
            for (i in 1:N){
                    p[i] = inv_logit(beta*x[i] + alpha);
                    }
            }
            model {
            alpha~normal(0,100);
            y~binomial(5,p);
            }
        """
        return log_reg