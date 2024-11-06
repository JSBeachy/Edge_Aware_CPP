import math


tot_length=500
probe_width=75

tot_passes=math.ceil(tot_length/probe_width)
per_pass_width=round(tot_length/tot_passes,2)

#print(per_pass_width)
#print(probe_width/2)

# #setting no edge
# benchmark=[]
# for i in range(tot_passes):
#     if i==0:
#         benchmark.append(per_pass_width/2)
#     else:
#         benchmark.append(benchmark[i-1]+per_pass_width)
#     #print(i)
#     #print(benchmark[i])

# print(benchmark)
# print(f"Distance to last edge: {tot_length-benchmark[-1]}")
# print(f"Amount hanging off the edge: {round(probe_width/2-benchmark[0],2)} mm")
# side_one= benchmark[1]-probe_width/2
# side_zero= benchmark[0]+probe_width/2
# overlap=side_zero-side_one
# #print(overlap)
# print(f"Percent reduncant coverage of interiror passes: {round(100*overlap*2/probe_width,2)}%")

# #setting one edge
# benchmark=[]
# for i in range(tot_passes):
#     if i==0:
#         benchmark.append(probe_width/2)
#     else:
#         benchmark.append(benchmark[i-1]+per_pass_width)
#     #print(i)
#     #print(benchmark[i])

# print(benchmark)
# print(tot_length-benchmark[-1])

#setting both edges

offset_one=probe_width/2
offset_two=tot_length-offset_one
benchmark=[offset_one]
real_distance=tot_length-2*probe_width
passes_left=math.ceil(real_distance/probe_width)
print(tot_passes)
print(passes_left)

for i in range(passes_left):
    if i==0:
        benchmark.append(benchmark[0]+per_pass_width/2)
    else:
        benchmark.append(benchmark[i-1]+per_pass_width)
    #print(i)
    #print(benchmark[i])
benchmark.append(offset_two)
print(benchmark)
print(tot_length-benchmark[-1])