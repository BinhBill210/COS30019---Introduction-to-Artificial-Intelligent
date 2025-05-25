# Traffic-Based Route Guidance System Test Report

*Generated on 2025-05-25 19:53:29*

## Summary

- Total tests: 12
- Passed: 8
- Failed: 4
- Errors: 0
- Time taken: 3.13 seconds

## Failures

### __main__.ModelTests.test_model_structure_bidirectional_lstm

```
Traceback (most recent call last):
  File "c:\Users\thaib\OneDrive - Swinburne University\Documents\Swinburne Major\COS30019 - Introduction to Aritificial Intelligent\Assignment 2\2B\test.py", line 280, in test_model_structure_bidirectional_lstm
    self.assertEqual(len(model.layers), 2, "Bidirectional LSTM model should have 2 layers")
AssertionError: 5 != 2 : Bidirectional LSTM model should have 2 layers

```

### __main__.ModelTests.test_model_structure_gru

```
Traceback (most recent call last):
  File "c:\Users\thaib\OneDrive - Swinburne University\Documents\Swinburne Major\COS30019 - Introduction to Aritificial Intelligent\Assignment 2\2B\test.py", line 268, in test_model_structure_gru
    self.assertEqual(len(model.layers), 2, "GRU model should have 2 layers")
AssertionError: 5 != 2 : GRU model should have 2 layers

```

### __main__.ModelTests.test_model_structure_lstm

```
Traceback (most recent call last):
  File "c:\Users\thaib\OneDrive - Swinburne University\Documents\Swinburne Major\COS30019 - Introduction to Aritificial Intelligent\Assignment 2\2B\test.py", line 256, in test_model_structure_lstm
    self.assertEqual(len(model.layers), 2, "LSTM model should have 2 layers")
AssertionError: 5 != 2 : LSTM model should have 2 layers

```

### __main__.ModelTests.test_travel_time_estimation

```
Traceback (most recent call last):
  File "c:\Users\thaib\OneDrive - Swinburne University\Documents\Swinburne Major\COS30019 - Introduction to Aritificial Intelligent\Assignment 2\2B\test.py", line 365, in test_travel_time_estimation
    self.assertLessEqual(time, tc["expected_max"],
AssertionError: np.float64(6.654456840426409) not less than or equal to 3.0 : Travel time 6.654456840426409 too high for flow 1000

```

