# ==============================================================================
# ===== CORE SIMULATION FUNCTION (UPDATED)
# ==============================================================================

def run_simulation_for_scenario(p_pi, p_mu, settings, tests_to_run):
    """
    Runs a full Monte Carlo simulation for a single (pi, mu) scenario,
    executing a provided dictionary of test functions.
    
    UPDATED: This version calculates the worst-case error by testing all
    three hypotheses (k=0, 1, 2 as the outlier) and taking the maximum
    error probability (Pe = max(pe1, pe2, pe3)).
    """
    lengths, trials, n_seqs = settings['lengths'], settings['trials'], settings['num_sequences']
    
    # This will store the final max Pe for each test
    errors = {key: [] for key in tests_to_run.keys()}

    for n in lengths:
        # This will temporarily store [pe1, pe2, pe3] for each test, for this length n
        hypothesis_errors = {key: [] for key in errors}
        print(f"    Running n={n:<4} (H1, H2, H3)...", end="", flush=True)
        
        # ### <<< NEW BLOCK >>> ###
        # Loop over each possible ground truth hypothesis (k = 0, 1, or 2)
        for k in range(n_seqs):
            # Reset error count for this specific hypothesis
            err_counts = {key: 0 for key in errors}
            
            for _ in range(trials):
                # ### <<< CHANGED LINE >>> ###
                # The ground truth is that the k-th sequence is the outlier
                seqs = [np.random.binomial(1, p_mu if i==k else p_pi, n) for i in range(n_seqs)]
                
                # Run each test and check for errors
                for test_key, test_info in tests_to_run.items():
                    decision = test_info['func'](seqs, **test_info['kwargs'])
                    
                    # ### <<< CHANGED LINE >>> ###
                    # An error occurs if the chosen index is not k
                    if decision != k: 
                        err_counts[test_key] += 1
            
            # ### <<< NEW BLOCK >>> ###
            # After all trials for hypothesis k, calculate its error rate (e.g., pe1)
            for key in errors:
                pe_k = err_counts[key] / trials
                hypothesis_errors[key].append(pe_k)
        
        # ### <<< NEW BLOCK >>> ###
        # Now that we have [pe1, pe2, pe3] for each test, find the max
        print(" Done.")
        for key in errors:
            pe_max = max(hypothesis_errors[key])
            errors[key].append(pe_max)
        
    df = pd.DataFrame({'Sequence_Length': lengths})
    for test_key, test_name in zip(tests_to_run.keys(), settings['tests'].keys()):
        df[f'Prob_Error_{test_name}'] = errors[test_key]
    return df