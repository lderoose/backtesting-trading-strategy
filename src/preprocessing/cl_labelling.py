#!/usr/bin/python3.8


import numpy as np
import pandas as pd


class Labelling:
    
    @staticmethod
    def _compute_threshold(stable_to_crypto, fees):
        """
        compute threshold for going stable to crypto or crypto to stable given fees.
        Parameters
        ----------
        stable_to_crypto : bool
        fees : float
            transaction fees
        Returns
        -------
        float
        """
        if stable_to_crypto:
            thresh = 1 / (1 - fees)**2
        else:
            thresh =  (1 - fees)**2 
        return thresh
    
    @staticmethod
    def compute_labels(streak_rdt, t0, T, fees):
        """
        given a series of expected return from a pair X/Y will create 2 vectors of labels :
            - first : label 1 we must sell X for Y, label 0 we must do nothing (don't move) 
            - second : label 1 we must sell Y for X, label 0 we must do nothing (don't move) 
        These labels are obtained by looking at future data, so the function also return 2 vectors called index_knowledge which notice at what time (in the future) 
        the information was found.
        Finally, 2 vectors of wheights for each labels are also provided as cumprod of expected return since t to t+h, h belongs to [1,T].
        In brief:
            first vector of labels answers to the question, if I am long on X at time t : should I buy Y or not? And this for each t belongs to [t0,T].
            second vector of labels answers to the question, if I am long on Y at time t : should I buy X or not? And this for each t belongs to [t0,T].
        Parameters
        ----------
        streak_rdt : numpy.ndarray
            expected return
        t0 : int
            start
        T : int
            end
        fees : float
            transaction fees
        Returns
        -------
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
            labelsXtoY, labelsYtoX, weightsXtoY, weightsYtoX, index_knowledgeXtoY, index_knowledgeYtoX
        """
        actions_crypto_to_stable = np.full(shape=len(streak_rdt[t0:T]), fill_value=2, dtype=np.uint8)
        actions_stable_to_crypto = np.full(shape=len(streak_rdt[t0:T]), fill_value=2, dtype=np.uint8)
        weights_crypto_to_stable = np.empty(shape=len(streak_rdt[t0:T]))
        weights_stable_to_crypto = np.empty(shape=len(streak_rdt[t0:T]))
        index_knowledge_crypto_to_stable = np.zeros(shape=len(streak_rdt[t0:T]), dtype=np.uint8)
        index_knowledge_stable_to_crypto = np.zeros(shape=len(streak_rdt[t0:T]), dtype=np.uint8)

        idx = np.arange(len(streak_rdt[t0:T]))
        cumprod_rdt = np.cumprod(1 + streak_rdt[t0:T])
        
        thresh_stable_to_crypto = __class__._compute_threshold(stable_to_crypto=True, fees=fees)
        thresh_crypto_to_stable = __class__._compute_threshold(stable_to_crypto=False, fees=fees)
        
        for i in range(len(streak_rdt[t0:T-1])):
            
            cumprod_sliced = cumprod_rdt[i+1:T] / cumprod_rdt[i]
            idx_sliced = idx[i+1:T] - (i+1)  # - (i+1) for starting at 0.
            
            # m1 : find the first time both conditions will met.
            argmin_c1 = next((x[0] for x in zip(idx_sliced, cumprod_sliced) if x[1] > thresh_stable_to_crypto), T - (t0 + i + 2))
            argmin_c2 = next((x[0] for x in zip(idx_sliced[:argmin_c1], cumprod_sliced[:argmin_c1]) if x[1] < 1), T - (t0 + i + 2))
            
            # which condition occur first ?
            is_c1_occur_first = True if argmin_c1 < argmin_c2 else False
            
            if is_c1_occur_first:
                actions_stable_to_crypto[i] = 1
                weights_stable_to_crypto[i] = cumprod_sliced[argmin_c1]
                index_knowledge_stable_to_crypto[i] = i + argmin_c1 + 1
            else:
                actions_stable_to_crypto[i] = 0
                weights_stable_to_crypto[i] = cumprod_sliced[argmin_c2]
                index_knowledge_stable_to_crypto[i] = i + argmin_c2 + 1
            
            # m2 : find the first time both conditions will met.
            argmin_c1 = next((x[0] for x in zip(idx_sliced, cumprod_sliced) if x[1] < thresh_crypto_to_stable), T - (t0 + i + 2))
            argmin_c2 = next((x[0] for x in zip(idx_sliced[:argmin_c1], cumprod_sliced[:argmin_c1]) if x[1] > 1), T - (t0 + i + 2))

            # which condition occur first ?
            is_c1_occur_first = True if argmin_c1 < argmin_c2 else False
            
            if is_c1_occur_first:
                actions_crypto_to_stable[i] = 1
                weights_crypto_to_stable[i] = cumprod_sliced[argmin_c1]
                index_knowledge_crypto_to_stable[i] = i + argmin_c1 + 1
            else:
                actions_crypto_to_stable[i] = 0
                weights_crypto_to_stable[i] = cumprod_sliced[argmin_c2]
                index_knowledge_crypto_to_stable[i] = i + argmin_c2 + 1
                
        # fill last value, unknown because no access to future
        actions_stable_to_crypto[-1], actions_crypto_to_stable[-1], weights_stable_to_crypto[-1], weights_crypto_to_stable[-1] = 0, 0, 0, 0
        
        return actions_stable_to_crypto, actions_crypto_to_stable, weights_stable_to_crypto, weights_crypto_to_stable, index_knowledge_stable_to_crypto,\
               index_knowledge_crypto_to_stable
               
    @staticmethod
    def compute_optimal_actions(streak_rdt, t0, T, fees, position_start=0):
        """
        given a vector of expected return will compute for each time if we must switch position or not. 
        action = 1 -> stay on crypto or buy crypto (sell stable).
        action = 0 -> stay on stable or buy stable (sell crypto).
        NB: action(t) depends to action(t-1) because action(t) <=> position(t+1), unlike method compute_labels.
        Parameters
        ----------
        streak_rdt : numpy.ndarray
            expected return
        t0 : int
            start
        T : int
            end
        fees : float
            transaction fees
        position_start : int, optional
            by default 0
        Returns
        -------
        numpy.ndarray
            for each time, the action that maximizes portfolio.
        """
        current_position = position_start
        optimal_actions = np.full(shape=len(streak_rdt[t0:T]), fill_value=2, dtype=np.uint8)
        idx = np.arange(len(streak_rdt[t0:T]))
        
        cumprod_rdt = np.cumprod(1 + streak_rdt[t0:T])
        
        thresh_stable_to_crypto = __class__._compute_threshold(stable_to_crypto=True, fees=fees)
        thresh_crypto_to_stable = __class__._compute_threshold(stable_to_crypto=False, fees=fees)
        
        for i in range(len(streak_rdt[t0:T-1])):
            
            cumprod_sliced = cumprod_rdt[i+1:T] / cumprod_rdt[i]
            idx_sliced = idx[i+1:T] - (i+1)  # - (i+1) for starting at 0.
            
            # long stable
            if not current_position:
                # m1 : find the first time both conditions will met.
                argmin_c1 = next((x[0] for x in zip(idx_sliced, cumprod_sliced) if x[1] > thresh_stable_to_crypto), T - (t0 + i + 2))
                argmin_c2 = next((x[0] for x in zip(idx_sliced[:argmin_c1], cumprod_sliced[:argmin_c1]) if x[1] < 1), T - (t0 + i + 2))

                # which condition occur first ?
                is_c1_occur_first = True if argmin_c1 < argmin_c2 else False

                if is_c1_occur_first:
                    optimal_actions[i] = 1
                else:
                    optimal_actions[i] = 0
                    
            # long crypto
            else:
                # m2 : find the first time both conditions will met.
                argmin_c1 = next((x[0] for x in zip(idx_sliced, cumprod_sliced) if x[1] < thresh_crypto_to_stable), T - (t0 + i + 2))
                argmin_c2 = next((x[0] for x in zip(idx_sliced[:argmin_c1], cumprod_sliced[:argmin_c1]) if x[1] > 1), T - (t0 + i + 2))

                # which condition occur first ?
                is_c1_occur_first = True if argmin_c1 < argmin_c2 else False

                if is_c1_occur_first:
                    optimal_actions[i] = 0
                else:
                    optimal_actions[i] = 1
            
            current_position = optimal_actions[i]
                
        # fill last value, unknown because no access to future
        optimal_actions[-1] = 0
        return optimal_actions
    
    
if __name__ ==  "__main__":
    # example of use and quick test
    # settings
    path_data = "data/preprocessed/preproc_BTCUSDT_1h_f0.00075_2017-09-28_02:00:00_to_2021-01-31_20:00:00.csv"
    
    # run
    df = pd.read_csv(path_data, index_col=0)
    optimal_actions = Labelling.compute_optimal_actions(streak_rdt=df.rdt_crypto_1h.values, t0=0, T=len(df.rdt_crypto_1h), fees=0.00075)
    labels_weights_index = Labelling.compute_labels(streak_rdt=df.rdt_crypto_1h.values, t0=0, T=len(df.rdt_crypto_1h), fees=0.00075)
    print(f"optimal_actions: {optimal_actions}")
    print(f"labels_stable_to_crypto: {labels_weights_index[0]}")
    print(f"labels_crypto_to_stable: {labels_weights_index[1]}")
    print(f"weights_stable_to_crypto: {labels_weights_index[2]}")
    print(f"weights_crypto_to_stable: {labels_weights_index[3]}")
    print(f"index_knowledge_stable_to_crypto: {labels_weights_index[4]}")
    print(f"index_knowledge_crypto_to_stable: {labels_weights_index[5]}")
