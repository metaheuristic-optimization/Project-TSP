### Project Structure

./dataset includes the test data files inst-0.tsp, inst-13.tsp inst-16.tsp
./experiments contain all experimental data such as raw outputs from test runs and openlibre excel documents with tabular results and graphs

### How to run

```
python TSP_toStudents.py <filename>
```

#### Example

```
python TSP_toStudents.py ./dataset/inst-0.tsp
```

You will be presented with the following 3 questions which require an integer value

#### Example of answering the input questions

```
Please select candidate selection method
1) Random selection
2) Roulette Wheel
3) Best and second best
1
Please select crossover method
1) Uniform crossover
2) Cycle crossover
2
Please select mutation method
1) Scramble mutation
2) Reciprocal exchange mutation
1
```

Please make sure to provide a correct integer value as there is no validation

