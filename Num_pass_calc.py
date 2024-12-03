import math


tot_length=500
probe_width=75

tot_passes=math.ceil(tot_length/probe_width)
per_pass_width=round(tot_length/tot_passes,2)

print(f"Probe width: {probe_width}")
print(f"Per-pass width: {per_pass_width}")

# Setting no edges

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
# side_one= benchmark[2]-probe_width/2
# side_zero= benchmark[1]+probe_width/2
# overlap=side_zero-side_one
# #print(overlap)
# print(f"Percent reduncant coverage of interiror passes: {round(100*overlap*2/probe_width,2)}%")


# Setting one edge

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

# Setting both edges
offset_one=probe_width/2
offset_two=tot_length-offset_one

benchmark=[]
interprolation=[]
real_distance=tot_length-2*probe_width
passes_left=math.ceil(real_distance/probe_width)
real_per_pass_width=real_distance/passes_left
print(f"Number of total passes: {tot_passes}")
print(f"Number of passes remaining after setting edges: {passes_left}")
print(f"New pass width after setting edges: {real_per_pass_width}")

#Option_1
interp_one=probe_width+real_per_pass_width/2
interp_two=(tot_length-probe_width)-real_per_pass_width/2
print(interp_two)
print(passes_left)
for i in range(tot_passes-1):
    if i==0:
        benchmark.append(offset_one)
        interprolation.append(offset_one)
    elif i==1:
        benchmark.append(probe_width+real_per_pass_width/2)
        interprolation.append(interp_one*(1-(i-1)/(passes_left-1))+interp_two*((i-1)/(passes_left-1)))
    else:
        benchmark.append(benchmark[i-1]+real_per_pass_width)
        interprolation.append(interp_one*(1-(i-1)/(passes_left-1))+interp_two*((i-1)/(passes_left-1)))
benchmark.append(offset_two)
interprolation.append(offset_two)

# Option_2
interp2=[]
for i in range(tot_passes):
    if i==0:
        interp2.append(offset_one)
    elif i==tot_passes-1:
        interp2.append(offset_two)
    else:
        interp2.append(interp_one*(1-(i-1)/(passes_left-1))+interp_two*((i-1)/(passes_left-1)))

#Option_3 Spreads out overlap a bit more, between interior and exteior passes(good compromise)
interp3=[]
for i in range(tot_passes):
    interp3.append(offset_one*(1-(i)/(tot_passes-1))+offset_two*((i)/(tot_passes-1)))

print(interp3)
print(interp2)
print(interprolation)
print(benchmark)
# print(f"Distance to last edge: {tot_length-benchmark[-1]}")
# print(f"Amount hanging off the edge: {round(probe_width/2-benchmark[0],2)} mm")
# side_one= benchmark[2]-probe_width/2
# side_zero= benchmark[1]+probe_width/2
# overlap=side_zero-side_one
# #print(overlap)
# print(f"Percent reduncant coverage of interiror passes: {round(100*overlap*2/probe_width,2)}%")



