# NFL Predictor 

This program predicts the most likely winner of NFL games. 

## Getting Started

The nfl_predictor.py program and all the data you need to run it with are included in the directory ML_Project_Aaron_Williams. 

### Prerequisites

Ensure that you have Python 3.6 properly installed on your machine. 

```
python --version
Python 3.6
```

## Replicate experiments

To replicate my experiments use the following command in the terminal. 

```
python3 nfl_predictor.py
``` 

### Expected output 

```
   ***** Validation Data 2014 ******    

    Week   Home_Team Away_Team  Prob_Home  Prob_Away Prediction Actual
0     16       49ers  Chargers  30.247095  69.719189       Away   Away
1     16       Bears     Lions  14.316929  85.659269       Away   Away
2     16     Bengals   Broncos  45.551386  53.790160       Away   Home
3     16  Buccaneers   Packers  13.891721  86.104911       Away   Away
4     16   Cardinals  Seahawks  44.116419  49.277921       Away   Away
5     16     Cowboys     Colts  45.610860  54.370114       Away   Home
6     16    Dolphins   Vikings  66.678318  33.293998       Home   Home
7     16     Jaguars    Titans  48.341929  51.657322       Away   Home
8     16        Jets  Patriots   8.326127  91.672249       Away   Away
9     16    Panthers    Browns  36.462721  63.529296       Away   Home
10    16     Raiders     Bills  12.079586  87.917206       Away   Home
11    16        Rams    Giants  64.344657  35.649984       Home   Away
12    16    Redskins    Eagles  13.855204  86.144231       Away   Home
13    16      Saints   Falcons  63.904565  36.093903       Home   Away
14    16    Steelers    Chiefs  67.545289  32.298136       Home   Home
15    16      Texans    Ravens  36.796119  62.921647       Away   Home

         Validation data NFL 2014 Week 16  Precision: 50.000%
         Validation data NFL 2014 Week 16  Recall: 22.222%
         Validation data NFL 2014 Week 16  Accuracy: 43.750%
         Validation data NFL 2014 Week 16  F-1 Measure: 30.769%

   ***** Test Data 2014 ******    

    Week   Home_Team  Away_Team  Prob_Home  Prob_Away Prediction Actual
0     17       49ers  Cardinals  29.429163  70.384658       Away   Home
1     17     Broncos    Raiders  93.176300   6.817983       Home   Home
2     17  Buccaneers     Saints  25.904245  74.095710       Away   Away
3     17      Chiefs   Chargers  38.663791  61.154807       Away   Home
4     17    Dolphins       Jets  84.409672  15.588092       Home   Away
5     17     Falcons   Panthers  46.268983  53.668689       Away   Away
6     17      Giants     Eagles  45.275221  54.717840       Away   Away
7     17     Packers      Lions  50.016889  49.644035       Home   Home
8     17    Patriots      Bills  81.482452  17.905455       Home   Away
9     17      Ravens     Browns  70.485339  29.407881       Home   Home
10    17    Redskins    Cowboys   7.804644  92.191101       Away   Away
11    17    Seahawks       Rams  82.067298  17.916282       Home   Home
12    17    Steelers    Bengals  38.239144  61.565033       Away   Home
13    17      Texans    Jaguars  89.919942   9.959119       Home   Home
14    17      Titans      Colts  13.017656  86.981774       Away   Away
15    17     Vikings      Bears  50.667906  49.322393       Home   Home

         Test data NFL 2014 Week 17  Precision: 75.000%
         Test data NFL 2014 Week 17  Recall: 66.667%
         Test data NFL 2014 Week 17  Accuracy: 68.750%
         Test data NFL 2014 Week 17  F-1 Measure: 70.588%

   ***** BASELINE RESULTS ******    

    Week   Home_Team  Away_Team Prediction Actual
0     17       49ers  Cardinals       Home   Home
1     17     Broncos    Raiders       Home   Home
2     17  Buccaneers     Saints       Home   Away
3     17      Chiefs   Chargers       Home   Home
4     17    Dolphins       Jets       Home   Away
5     17     Falcons   Panthers       Home   Away
6     17      Giants     Eagles       Home   Away
7     17     Packers      Lions       Home   Home
8     17    Patriots      Bills       Home   Away
9     17      Ravens     Browns       Home   Home
10    17    Redskins    Cowboys       Home   Away
11    17    Seahawks       Rams       Home   Home
12    17    Steelers    Bengals       Home   Home
13    17      Texans    Jaguars       Home   Home
14    17      Titans      Colts       Home   Away
15    17     Vikings      Bears       Home   Home

         Baseline NFL 2014 Week 17  Precision: 56.250%
         Baseline NFL 2014 Week 17  Recall: 100.000%
         Baseline NFL 2014 Week 17  Accuracy: 56.250%
         Baseline NFL 2014 Week 17  F-1 Measure: 72.000%

   ***** Validation Data 2015 ******    

    Week   Home_Team Away_Team  Prob_Home  Prob_Away Prediction Actual
0     16       Bills   Cowboys  35.434182  64.557449       Away   Home
1     16     Broncos   Bengals  41.310312  58.642237       Away   Home
2     16  Buccaneers     Bears  48.175744  51.822248       Away   Away
3     16   Cardinals   Packers  62.014553  37.482622       Home   Home
4     16      Chiefs    Browns  83.320061  16.654640       Home   Home
5     16    Dolphins     Colts  39.960360  60.003964       Away   Away
6     16      Eagles  Redskins  69.296055  30.673128       Home   Away
7     16     Falcons  Panthers  23.713963  75.730327       Away   Home
8     16        Jets  Patriots  32.778080  67.147502       Away   Home
9     16       Lions     49ers  58.754707  41.238992       Home   Home
10    16     Raiders  Chargers  55.190628  44.808236       Home   Home
11    16      Ravens  Steelers  26.116805  73.882496       Away   Home
12    16      Saints   Jaguars  74.533709  25.464850       Home   Home
13    16    Seahawks      Rams  75.448425  24.434042       Home   Away
14    16      Titans    Texans  30.621992  69.376136       Away   Away
15    16     Vikings    Giants  70.495646  29.470854       Home   Home

         Validation data NFL 2015 Week 16  Precision: 75.000%
         Validation data NFL 2015 Week 16  Recall: 54.545%
         Validation data NFL 2015 Week 16  Accuracy: 56.250%
         Validation data NFL 2015 Week 16  F-1 Measure: 63.158%
         Best parameter for solver:  {'solver': 'liblinear'}

   ***** Test Data 2015 ******    

    Week  Home_Team   Away_Team  Prob_Home  Prob_Away Prediction Actual
0     17      49ers        Rams  39.829441  60.066934       Away   Home
1     17      Bears       Lions  52.868121  47.129703       Home   Away
2     17    Bengals      Ravens  80.961248  18.966698       Home   Home
3     17      Bills        Jets  52.745260  47.224740       Home   Home
4     17    Broncos    Chargers  80.045493  19.952063       Home   Home
5     17     Browns    Steelers  24.691410  75.307316       Away   Away
6     17  Cardinals    Seahawks  67.684900  32.247207       Home   Away
7     17     Chiefs     Raiders  81.411150  18.472499       Home   Home
8     17      Colts      Titans  76.476300  23.513879       Home   Home
9     17    Cowboys    Redskins  54.645965  45.347215       Home   Away
10    17   Dolphins    Patriots  19.001074  80.974802       Away   Home
11    17    Falcons      Saints  65.684227  34.312162       Home   Away
12    17     Giants      Eagles  46.609219  53.381864       Away   Away
13    17    Packers     Vikings  68.436296  30.879703       Home   Away
14    17   Panthers  Buccaneers  87.401378  12.577674       Home   Home
15    17     Texans     Jaguars  87.468968  12.527989       Home   Home

         Test data NFL 2015 Week 17  Precision: 58.333%
         Test data NFL 2015 Week 17  Recall: 77.778%
         Test data NFL 2015 Week 17  Accuracy: 56.250%
         Test data NFL 2015 Week 17  F-1 Measure: 66.667%
         Best parameter for solver:  {'solver': 'liblinear'}

   ***** BASELINE RESULTS ******    

    Week  Home_Team   Away_Team Prediction Actual
0     17      49ers        Rams       Home   Home
1     17      Bears       Lions       Home   Away
2     17    Bengals      Ravens       Home   Home
3     17      Bills        Jets       Home   Home
4     17    Broncos    Chargers       Home   Home
5     17     Browns    Steelers       Home   Away
6     17  Cardinals    Seahawks       Home   Away
7     17     Chiefs     Raiders       Home   Home
8     17      Colts      Titans       Home   Home
9     17    Cowboys    Redskins       Home   Away
10    17   Dolphins    Patriots       Home   Home
11    17    Falcons      Saints       Home   Away
12    17     Giants      Eagles       Home   Away
13    17    Packers     Vikings       Home   Away
14    17   Panthers  Buccaneers       Home   Home
15    17     Texans     Jaguars       Home   Home

         Baseline NFL 2015 Week 17  Precision: 56.250%
         Baseline NFL 2015 Week 17  Recall: 100.000%
         Baseline NFL 2015 Week 17  Accuracy: 56.250%
         Baseline NFL 2015 Week 17  F-1 Measure: 72.000%

   ***** Validation Data 2016 ******    

         C = .001 Accuracy: 66.667%
         C = .01 Accuracy: 66.667%
         C = .1 Accuracy: 60.000%
         C = 1 Accuracy: 46.667%
         C = 10 Accuracy: 46.667%
         C = 100 Accuracy: 46.667%
         C = 1000 Accuracy: 46.667%
         Best parameter for solver:  {'C': 0.001}

   ***** Test Data 2016 ******    

         C = .001 Accuracy: 75.000%
         C = .01 Accuracy: 68.750%
         C = .1 Accuracy: 75.000%
         C = 1 Accuracy: 75.000%
         C = 10 Accuracy: 75.000%
         C = 100 Accuracy: 75.000%
         C = 1000 Accuracy: 75.000%
         Best parameter for solver:  {'C': 0.001}

   ***** BASELINE RESULTS ******    

    Week Home_Team  Away_Team Prediction Actual
0     17     49ers   Seahawks       Home   Away
1     17     Bears      Lions       Home   Away
2     17   Bengals     Ravens       Home   Home
3     17   Broncos    Raiders       Home   Home
4     17  Chargers     Chiefs       Home   Away
5     17     Colts    Jaguars       Home   Home
6     17  Dolphins   Patriots       Home   Away
7     17    Eagles    Cowboys       Home   Home
8     17   Falcons     Saints       Home   Home
9     17      Jets      Bills       Home   Home
10    17     Lions    Packers       Home   Away
11    17      Rams  Cardinals       Home   Away
12    17  Redskins     Giants       Home   Away
13    17  Steelers     Browns       Home   Home
14    17    Titans     Texans       Home   Home
15    17   Vikings      Bears       Home   Home

         Baseline NFL 2016 Week 17  Precision: 56.250%
         Baseline NFL 2016 Week 17  Recall: 100.000%
         Baseline NFL 2016 Week 17  Accuracy: 56.250%
         Baseline NFL 2016 Week 17  F-1 Measure: 72.000%

   ***** Validation Data 2017 ******    

    Week Home_Team   Away_Team  Prob_Home  Prob_Away Prediction Actual
0     11   Broncos     Bengals  51.309152  47.589673       Home   Away
1     11    Browns     Jaguars  32.056980  67.695070       Away   Away
2     11  Chargers       Bills  56.517677  42.935640       Home   Home
3     11   Cowboys      Eagles  45.396616  54.326305       Away   Away
4     11  Dolphins  Buccaneers  47.266967  52.549440       Away   Away
5     11    Giants      Chiefs  34.678747  65.235640       Away   Home
6     11   Packers      Ravens  55.140862  44.433354       Home   Away
7     11   Raiders    Patriots  41.483426  58.416646       Away   Away
8     11    Saints    Redskins  71.228959  28.101906       Home   Home
9     11  Seahawks     Falcons  64.261854  35.338456       Home   Away
10    11  Steelers      Titans  65.885471  33.240650       Home   Home
11    11    Texans   Cardinals  51.551360  48.119765       Home   Home
12    11   Vikings        Rams  55.874989  43.737144       Home   Home

         Validation data NFL 2017 Week 10  Precision: 62.500%
         Validation data NFL 2017 Week 10  Recall: 83.333%
         Validation data NFL 2017 Week 10  Accuracy: 69.231%
         Validation data NFL 2017 Week 10  F-1 Measure: 71.429%
         Best parameter for intercept scaling:  {'penalty': 'l2'}

   ***** Test Data 2017 ******    

    Week  Home_Team   Away_Team  Prob_Home  Prob_Away Prediction Actual
0     12      49ers    Seahawks  37.869492  61.997855       Away   Away
1     12    Bengals      Browns  59.231768  40.412815       Home   Home
2     12  Cardinals     Jaguars  42.205412  57.573158       Away   Home
3     12     Chiefs       Bills  64.705729  34.908244       Home   Away
4     12      Colts      Titans  39.787692  60.056467       Away   Away
5     12    Cowboys    Chargers  53.889584  45.760784       Home   Away
6     12     Eagles       Bears  74.979843  24.040864       Home   Home
7     12    Falcons  Buccaneers  62.828266  36.690166       Home   Home
8     12       Jets    Panthers  45.684084  54.069136       Away   Away
9     12      Lions     Vikings  48.087433  51.730138       Away   Away
10    12   Patriots    Dolphins  78.019212  21.495535       Home   Home
11    12    Raiders     Broncos  55.058749  44.705350       Home   Home
12    12       Rams      Saints  56.096555  43.717749       Home   Home
13    12     Ravens      Texans  52.062475  47.671228       Home   Home
14    12   Redskins      Giants  59.766544  39.833998       Home   Home
15    12   Steelers     Packers  69.974706  29.005837       Home   Home

         Test data NFL 2017 Week 12  Precision: 81.818%
         Test data NFL 2017 Week 12  Recall: 90.000%
         Test data NFL 2017 Week 12  Accuracy: 81.250%
         Test data NFL 2017 Week 12  F-1 Measure: 85.714%
         Best parameter for intercept scaling:  {'penalty': 'l2'}

   ***** BASELINE RESULTS ******    

    Week  Home_Team   Away_Team Prediction Actual
0     12      49ers    Seahawks       Home   Away
1     12    Bengals      Browns       Home   Home
2     12  Cardinals     Jaguars       Home   Home
3     12     Chiefs       Bills       Home   Away
4     12      Colts      Titans       Home   Away
5     12    Cowboys    Chargers       Home   Away
6     12     Eagles       Bears       Home   Home
7     12    Falcons  Buccaneers       Home   Home
8     12       Jets    Panthers       Home   Away
9     12      Lions     Vikings       Home   Away
10    12   Patriots    Dolphins       Home   Home
11    12    Raiders     Broncos       Home   Home
12    12       Rams      Saints       Home   Home
13    12     Ravens      Texans       Home   Home
14    12   Redskins      Giants       Home   Home
15    12   Steelers     Packers       Home   Home

         Baseline NFL 2017 Week 12  Precision: 62.500%
         Baseline NFL 2017 Week 12  Recall: 100.000%
         Baseline NFL 2017 Week 12  Accuracy: 62.500%
         Baseline NFL 2017 Week 12  F-1 Measure: 76.923%

```

## Built With

* [Pycharm Community Edition 4.4.5](https://www.jetbrains.com/pycharm/) - The IDE used

## Authors

* Aaron Williams ** 

