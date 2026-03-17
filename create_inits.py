import os

dirs = [
    'models', 'models/supervised_learning', 'models/ensemble_learning',
    'models/unsupervised_learning', 'models/semi_supervised_learning',
    'models/deep_learning', 'models/graph_neural_network',
    'models/probabilistic_graphical_model', 'models/large_language_model',
    'models/time_series', 'models/reinforcement_learning',
    'models/nlp', 'models/computer_vision',
    'models/anomaly_detection', 'models/causal_inference'
]
for d in dirs:
    init = os.path.join(d, '__init__.py')
    if not os.path.exists(init):
        open(init, 'w').close()
        print(f'Created: {init}')
    else:
        print(f'Exists:  {init}')
