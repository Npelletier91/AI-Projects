import numpy as np
import skfuzzy as fuzz


x = np.arange(0,20,1)
#print(x)

low = fuzz.trimf(x, [0,0,5])
medium = fuzz.trimf(x, [2,5,8])
high = fuzz.trimf(x,[5,10,10])
custom = fuzz.trimf(x, [3,6,10])

rule1 = np.fmax(low,medium)
rule2 = np.fmin(medium,high)

relation = np.fmax(rule1,rule2)

output = fuzz.defuzz(x, relation, 'centroid')

print("Output:", output)
print("LOw:",low)
print("medium:", medium)
print("high", high)
print("custom", custom)
print("relation:", relation)
