#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:07:21 2025

@author: dabdelkader
"""
from Optimizer.Optimizer import Optimizer
from Utilities.BaseUtility import BaseUtility
import os
import numpy as np
import pickle 
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#TODO: I need to add an optimizer that just get the gradient and then the step,
# this could use adam or other ML optimizers
_optimizer_registry = {}
scipy_methods = [ 'Nelder-Mead','Powell', 'CG',  'BFGS', 'Newton-CG','L-BFGS-B',
                 'TNC', 'COBYLA','COBYQA','SLSQP','trust-constr','dogleg','trust-ncg',
                 'trust-exact','trust-krylov']


def register_optimizer(name):
    def decorator(cls):
        if name.lower()=="scipy":
            for method in scipy_methods:                
                _optimizer_registry[method.lower()] = cls
        else:
            _optimizer_registry[name.lower()] = cls
        return cls
    return decorator

def get_optimizer(parsed_args, *args, **kwargs) -> Optimizer:
    method=parsed_args.optimizer    
    if method.lower() not in _optimizer_registry:
        raise ValueError(f"Unsupported optimizer: {method}. Supported: {_optimizer_registry.keys()}")
    
    optimizer = _optimizer_registry[method.lower()](parsed_args, *args, **kwargs)
    if method in scipy_methods:
        optimizer.set_method(method)
        
    return optimizer



@register_optimizer("random")
class RandomOptimizer(Optimizer):
    def optimize(self):
        from scipy.optimize import dual_annealing

        logger.info("Running Random Search via Dual Annealing...")
        x0 = [v for k,v in self.initial_values.items()]
        result = dual_annealing(self._objective, 
                                bounds=list(zip(self.lb, self.ub)), 
                                maxiter=self.max_evals,
                                x0 = x0)
        best_params = dict(zip(self.param_names, result.x))
        best_value = result.fun
        return {"params": best_params, "loss": best_value}


@register_optimizer("bayesian")
class BayesianOptimizer(Optimizer):
    def optimize(self):
        from skopt import gp_minimize
        from skopt.space import Real

        logger.info("Running Bayesian Optimization...")
        
        dimensions = [Real(lb, ub) for lb, ub in zip(self.lb, self.ub)]
        res = gp_minimize(
            func=self._objective,
            dimensions=dimensions,
            n_calls=self.max_evals,
            initial_point_generator = "lhs",
            n_initial_points = int(self.max_evals/5), #use 10%
        )

        best_params = dict(zip(self.param_names, res.x))
        return {"params": best_params, "loss": res.fun}



@register_optimizer("tpe")
class TPEOptimizer(Optimizer):
    def optimize(self):
        from hyperopt import fmin, tpe, hp, Trials

        logger.info("Running TPE Optimization...")

        space = {name: hp.uniform(name, lb, ub) for name, lb, ub in zip(self.param_names, self.lb, self.ub)}

        def objective(args):
            x = [args[name] for name in self.param_names]
            return self._objective(x)

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=self.max_evals, 
                    trials=trials, show_progressbar = False, verbose = False)
        return {"params": best, "loss": trials.best_trial['result']['loss']}

    
@register_optimizer("pso")
class PSOOptimizer(Optimizer):
    def optimize(self):
        import pyswarms as ps
        import numpy as np

        logger.info("Running Particle Swarm Optimization...")

        bounds = (self.lb, self.ub)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        # Wrap the objective function to work with a 2D array input
        def vectorized_objective(X):
            return np.array([self._objective(x) for x in X])

        optimizer = ps.single.GlobalBestPSO(
            n_particles=10, 
            dimensions=len(self.param_names),
            options=options, 
            bounds=bounds
        )
        cost, pos = optimizer.optimize(vectorized_objective, iters=self.max_evals // 10)

        return {"params": dict(zip(self.param_names, pos)), "loss": cost}



@register_optimizer("ga")
class GAOptimizer(Optimizer):
    def optimize(self):
        import random
        from deap import base, creator, tools, algorithms
        import numpy as np

        logger.info("Running Genetic Algorithm...")

        # Safely create types only once
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Attribute generators for each parameter
        for i, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            toolbox.register(f"x{i}", random.uniform, lb, ub)

        # Structure initializers
        toolbox.register("individual", tools.initCycle, creator.Individual,
                 tuple(getattr(toolbox, f"x{i}") for i in range(len(self.param_names))), n=1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function must return a tuple
        toolbox.register("evaluate", lambda ind: (self._objective(ind),))
        if len(self.param_names) > 1:
            toolbox.register("mate", tools.cxTwoPoint)
        else:
            toolbox.register("mate", tools.cxUniform, indpb=1.0)  # for 1D individuals

        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=24)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)

        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1,
                            ngen=self.max_evals // 20,
                            stats=stats, halloffame=hof, verbose=False)

        return {"params": dict(zip(self.param_names, hof[0])), "loss": hof[0].fitness.values[0]}


@register_optimizer("spsa")
class SPSAOptimizer(Optimizer):
    def optimize(self):
        import numpy as np
        import spsa

        logger.info("Running SPSA Optimization using spsa.minimize...")

        dim = len(self.param_names)
        x0 =  [v for k,v in self.initial_values.items()]

        def wrapped_objective(x):
            x = np.clip(x, self.lb, self.ub)
            return self._objective(x)
        
        x_best = spsa.minimize(wrapped_objective,
                               x0,
                               iterations=self.max_evals)
        
        loss_best = wrapped_objective(x_best)

        return {
            "params": dict(zip(self.param_names, x_best.tolist())),
            "loss": loss_best
        }

    

    
@register_optimizer("adam")
class FiniteDifferenceAdamOptimizer(Optimizer):
    def optimize(self):
        import numpy as np
        from tqdm import tqdm
        logger.info("Running Adam Optimization with Finite-Difference Gradient Estimation...")

        # Adam parameters
        alpha = 0.1  # learning rate
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        dim = len(self.param_names)
        x = np.array([(low + high) / 2 for low, high in zip(self.lb, self.ub)])  # Initial guess
        m = np.zeros(dim)
        v = np.zeros(dim)

        best_loss = float('inf')
        best_x = x.copy()
        no_improvement = 0
        
        def estimate_grad(f, x, epsilon=0.5):
            grad = np.zeros_like(x)
            fx = f(x)
            for i in range(len(x)):
                x_step = x.copy()
                x_step[i] += epsilon
                grad[i] = (f(x_step) - fx) / epsilon
            return grad

        for t in tqdm(range(1, self.max_evals + 1)):
            grad = estimate_grad(self._objective, x)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            x -= alpha * m_hat / (np.sqrt(v_hat) + eps)

            # Project back into bounds
            x = np.clip(x, self.lb, self.ub)

            loss = self._objective(x)
            if loss < best_loss:
                best_loss = loss
                best_x = x.copy()
                no_improvement = 0
            else:
               no_improvement+=1 
            
            if no_improvement>10:
                logger.info("Breaking the loop after 10 iterations without improvements")
                break

        return {
            "params": dict(zip(self.param_names, best_x.tolist())),
            "loss": best_loss
        }




@register_optimizer("kai")
class KaiOptimizer(Optimizer):
    def optimize(self):
        import numpy as np        
        logger.info("Using Kai utility calibration formula...")

        modes = ["pt", "car", "walk", "bike", "car_passenger"]
        actual_mode_shares = self.get_actual_mode_shares(modes)["global"]
        reference_mode = self.get_reference_mode()
        logger.info(f"The reference mode used in Kai optimizer is: {reference_mode}")

        max_iter = 20
        tol = 1e-3
        prev_mode_shares = None
        optimal_params = None
        loss_function = self.objective_function.mse()
        
        for i in range(max_iter):
            simulated_mode_shares = self.get_estimated_mode_shares(modes)["global"]

            # Compute difference for convergence
            diff = np.linalg.norm(
                np.array([simulated_mode_shares[m] for m in modes if m in simulated_mode_shares]) -
                np.array([prev_mode_shares[m] for m in modes if prev_mode_shares and m in prev_mode_shares])
            ) if prev_mode_shares is not None else np.inf

            logger.info(f"KaiOptimizer (it.{i}): Change in mode shares: {diff:.6f}")

            prev_mode_shares = simulated_mode_shares

            # Update parameters
            optimal_params = self._one_iteration(
                simulated_mode_shares=simulated_mode_shares,
                actual_mode_shares=actual_mode_shares,
                iteration=i,
                reference_mode=reference_mode
            )
            BaseUtility.set_parameters(optimal_params)

            mse = loss_function(
                np.array([simulated_mode_shares[m] for m in modes if m in simulated_mode_shares]),
                np.array([actual_mode_shares[m] for m in modes if m in actual_mode_shares])
            )
            self.explored_solutions.append(list(optimal_params.values()))
            self.explored_objectives.append(mse)

            # Check for convergence at the end, because one more adjustment would give better results, and it is not expensive
            # the get_estimated_mode_shares is the one that is expensive
            if diff < tol:
                logger.info("Converged.")
                break

        return {"params": optimal_params, "loss": diff if optimal_params is not None else np.nan}

    def _one_iteration(self, simulated_mode_shares, actual_mode_shares, iteration, beta = 0.8, reference_mode = "pt"):        
        calibrated_modes = ["pt", "car", "walk", "bike"]        
        calibrated_modes.remove(reference_mode)
        
        params = [f"{mode}.alpha_u" for mode in calibrated_modes]
        
        initial_parameters_values = self.get_current_parameters(params)

        z0 = actual_mode_shares[reference_mode][0]  # Reference (e.g., pt)
        m0 = simulated_mode_shares[reference_mode][0]  # Simulated reference share

        zi = np.array([actual_mode_shares[i][0] for i in calibrated_modes])  # Others: car, walk, bike
        mi = np.array([simulated_mode_shares[i][0] for i in calibrated_modes])
        asci = np.array([initial_parameters_values[i] for i in params])

        # Update parameters using Kai's formula
        new_parameters_values = (
            asci +
            (np.log(zi) - np.log(mi)) -
            (np.log(z0) - np.log(m0))
        )
        
        beta = min(beta, 1-1/(0.5*iteration+1))
        new_parameters_values = beta*asci+(1-beta)*new_parameters_values        
        return dict(zip(params, new_parameters_values.tolist()))


    def get_reference_mode(self):
        all_modes = ["pt", "car", "walk", "bike"]
        params = [f"{mode}.alpha_u" for mode in all_modes]
        # reference mode is supposed to be the mode not present in the bounds
        keys = self.bounds.keys()
        reference_mode = list(set(params)-set(keys))
        assert len(reference_mode)==1, "If you are only calibrating the betas, there should be exactly one reference mode. All other alphas are adjusted."
        reference_mode = reference_mode[0].split('.')[0]
        assert reference_mode in all_modes, "Couldn't find the reference mode in Kai optimizer."
        return reference_mode
    
    

@register_optimizer("scipy")
class ScipyOptimizer(Optimizer):
    method = 'Nelder-Mead'
    def set_method(self, method):
        ScipyOptimizer.method = method
        
    def optimize(self):
        from scipy.optimize import minimize

        logger.info(f"Running {ScipyOptimizer.method} Optimization...")
        x0 = [v for k,v in self.initial_values.items()]
        res = minimize(self._objective, 
                       x0=x0, 
                       bounds=list(zip(self.lb, self.ub)),
                       method=ScipyOptimizer.method)
        return {"params": dict(zip(self.param_names, res.x)), "loss": res.fun}
        

@register_optimizer("dual_annealing")
class DualAnnealing(Optimizer):       
    def optimize(self):
        from scipy.optimize import dual_annealing

        logger.info(f"Running dual_annealing Optimization...")        
        res = dual_annealing(self._objective,                        
                             bounds=list(zip(self.lb, self.ub)),
                             maxfun = self.max_evals)
        
        return {"params": dict(zip(self.param_names, res.x)), "loss": res.fun}


@register_optimizer("cmaes")
class CMAESOptimizer(Optimizer):
    
    def start_new_optimizer(self, scaler, unscaler, lb, ub, more_evals = 0):
        import cma        
        x0 = np.array([v for k,v in self.initial_values.items()])
        x0_scaled = scaler(x0)   
        
        sigma = 0.3
        num_param = len(self.param_names)
        popsize = int(12+14*np.ceil(np.log(num_param)))
        logger.info(f"    Population size is set to {popsize}")
        
        #Now, use CMA-ES optimization
        options = cma.CMAOptions()
        options.set("bounds", [np.zeros_like(lb), np.ones_like(lb)])
        options.set("maxfevals", self.max_evals + more_evals)        
        options.set("popsize", popsize)
        options.set("tolfun", 1e-3)  # Stop if function value changes less than 1e-3
        options.set("tolx", 5e-3)    # Stop if parameters change less than 5e-3
        options.set("seed", 1102)
        es = cma.CMAEvolutionStrategy(x0_scaled, sigma, options)  
        return es

    def get_cmaes_optimizer(self, scaler, unscaler, lb, ub, overwrite):        
        # the file where the cached optimizer might be stored
        filename = self.cache_state_file    
        # Ensure explored_solutions and explored_objectives are non-empty lists before resuming
        new_optimizer = (
            not os.path.exists(filename)
            or len(self.explored_solutions) == 0
            or len(self.explored_objectives) == 0
            or overwrite
        )
        more_evals= 0 if new_optimizer else int(self.max_evals*0.5)

        es = self.start_new_optimizer(scaler, unscaler, lb, ub, more_evals=more_evals)           
       
        if not new_optimizer:
            es.feed_for_resume(self.explored_solutions, self.explored_objectives)
            if es.sigma<0.1:
                es.sigma = min(0.1, es.sigma*3)

            logger.info(f"Resuming CMA-ES optimization from {len(self.explored_solutions)} explored solutions.")
        else:
            self.explored_solutions, self.explored_objectives = [], []
            logger.info("Starting new CMA-ES optimization.")

        return es
        
    def optimize(self, overwrite = False):        
                       
        logger.info("Running CMA-ES Optimization...")
        # Load previous state if exists
        if not overwrite:
            self.load_state() 

        # Convert bounds to arrays and define scaling functions
        lb, ub = np.array(self.lb), np.array(self.ub)
        scaler = lambda x: (x - lb) / (ub - lb)
        unscaler = lambda x: x * (ub - lb) + lb
                 
        es = self.get_cmaes_optimizer(scaler, unscaler, lb, ub, overwrite)                     
        
        #iteration = 0        
        #while (not es.stop()) and (iteration<5): #at least 5 iterations
        for _ in range(10): # do 10 iterations each matsim iteration 
            solutions = es.ask()                                           
            objectives = [self._objective(unscaler(sol), sol) for sol in solutions]                        
            es.tell(solutions, objectives)                        
            es.disp()    
        #    iteration +=1            
        
        # Get the best solution among these run iterations
        num = es.popsize * 10
        recent_solutions = self.explored_solutions[-num:] if len(self.explored_solutions) >= num else None
        recent_objectives = self.explored_objectives[-num:] if len(self.explored_objectives) >= num else None

        if recent_solutions and recent_objectives:
            best_idx = np.argmin(recent_objectives)
            xbest = unscaler(recent_solutions[best_idx])
        else:
            xbest = unscaler(es.result.xbest)
        
        self.save_state()
        return {"params": dict(zip(self.param_names, xbest)), "loss": es.result.fbest}    
        


