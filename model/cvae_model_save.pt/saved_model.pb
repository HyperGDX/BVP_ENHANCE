
§
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8ÎÌ

time_distributed_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed_7/bias

+time_distributed_7/bias/Read/ReadVariableOpReadVariableOptime_distributed_7/bias*
_output_shapes
:*
dtype0

time_distributed_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nametime_distributed_7/kernel

-time_distributed_7/kernel/Read/ReadVariableOpReadVariableOptime_distributed_7/kernel*&
_output_shapes
:*
dtype0

time_distributed_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed_6/bias

+time_distributed_6/bias/Read/ReadVariableOpReadVariableOptime_distributed_6/bias*
_output_shapes
:*
dtype0

time_distributed_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nametime_distributed_6/kernel

-time_distributed_6/kernel/Read/ReadVariableOpReadVariableOptime_distributed_6/kernel*&
_output_shapes
: *
dtype0

time_distributed_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nametime_distributed_4/bias

+time_distributed_4/bias/Read/ReadVariableOpReadVariableOptime_distributed_4/bias*
_output_shapes	
:@*
dtype0

time_distributed_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 @**
shared_nametime_distributed_4/kernel

-time_distributed_4/kernel/Read/ReadVariableOpReadVariableOptime_distributed_4/kernel*
_output_shapes
:	 @*
dtype0

time_distributed_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nametime_distributed_3/bias

+time_distributed_3/bias/Read/ReadVariableOpReadVariableOptime_distributed_3/bias*
_output_shapes
:@*
dtype0

time_distributed_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@@**
shared_nametime_distributed_3/kernel

-time_distributed_3/kernel/Read/ReadVariableOpReadVariableOptime_distributed_3/kernel*
_output_shapes
:	@@*
dtype0

time_distributed_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametime_distributed_1/bias

+time_distributed_1/bias/Read/ReadVariableOpReadVariableOptime_distributed_1/bias*
_output_shapes
: *
dtype0

time_distributed_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nametime_distributed_1/kernel

-time_distributed_1/kernel/Read/ReadVariableOpReadVariableOptime_distributed_1/kernel*&
_output_shapes
: *
dtype0

time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nametime_distributed/bias
{
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes
:*
dtype0

time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed/kernel

+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
ª`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*å_
valueÛ_BØ_ BÑ_
I
	keras_api
encoder
decoder

sample

signatures*
* 

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	layer_with_weights-2
	layer-3

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	 layer*

!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
	'layer*

(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
	.layer* 

/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
	5layer*
.
60
71
82
93
:4
;5*
.
60
71
82
93
:4
;5*
* 

<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics

	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Atrace_0
Btrace_1
Ctrace_2
Dtrace_3* 
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 

I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
	Olayer*

P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
	Vlayer* 

W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
	]layer*

^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
	dlayer*
.
e0
f1
g2
h3
i4
j5*
.
e0
f1
g2
h3
i4
j5*
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
6
ttrace_0
utrace_1
vtrace_2
wtrace_3* 

60
71*

60
71*
* 

xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

}trace_0
~trace_1* 

trace_0
trace_1* 
Ï
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

6kernel
7bias
!_jit_compiled_convolution_op*

80
91*

80
91*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
Ï
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

8kernel
9bias
!_jit_compiled_convolution_op*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
 trace_1* 

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses* 

:0
;1*

:0
;1*
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

¬trace_0
­trace_1* 

®trace_0
¯trace_1* 
¬
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses

:kernel
;bias*
_Y
VARIABLE_VALUEtime_distributed/kernel.encoder/variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEtime_distributed/bias.encoder/variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEtime_distributed_1/kernel.encoder/variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtime_distributed_1/bias.encoder/variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEtime_distributed_3/kernel.encoder/variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtime_distributed_3/bias.encoder/variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
	3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

e0
f1*

e0
f1*
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

»trace_0
¼trace_1* 

½trace_0
¾trace_1* 
¬
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses

ekernel
fbias*
* 
* 
* 

Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

Êtrace_0
Ëtrace_1* 

Ìtrace_0
Ítrace_1* 

Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses* 

g0
h1*

g0
h1*
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

Ùtrace_0
Útrace_1* 

Ûtrace_0
Ütrace_1* 
Ï
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses

gkernel
hbias
!ã_jit_compiled_convolution_op*

i0
j1*

i0
j1*
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

étrace_0
êtrace_1* 

ëtrace_0
ìtrace_1* 
Ï
í	variables
îtrainable_variables
ïregularization_losses
ð	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses

ikernel
jbias
!ó_jit_compiled_convolution_op*
a[
VARIABLE_VALUEtime_distributed_4/kernel.decoder/variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtime_distributed_4/bias.decoder/variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEtime_distributed_6/kernel.decoder/variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtime_distributed_6/bias.decoder/variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEtime_distributed_7/kernel.decoder/variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtime_distributed_7/bias.decoder/variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

 0*
* 
* 
* 
* 
* 
* 
* 

60
71*

60
71*
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ùtrace_0* 

útrace_0* 
* 
* 

'0*
* 
* 
* 
* 
* 
* 
* 

80
91*

80
91*
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
	
.0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

50*
* 
* 
* 
* 
* 
* 
* 

:0
;1*

:0
;1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

O0*
* 
* 
* 
* 
* 
* 
* 

e0
f1*

e0
f1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
	
V0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

]0*
* 
* 
* 
* 
* 
* 
* 

g0
h1*

g0
h1*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses*

£trace_0* 

¤trace_0* 
* 
* 

d0*
* 
* 
* 
* 
* 
* 
* 

i0
j1*

i0
j1*
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
í	variables
îtrainable_variables
ïregularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses*

ªtrace_0* 

«trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ì
StatefulPartitionedCallStatefulPartitionedCallsaver_filename+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_1/kernel/Read/ReadVariableOp+time_distributed_1/bias/Read/ReadVariableOp-time_distributed_3/kernel/Read/ReadVariableOp+time_distributed_3/bias/Read/ReadVariableOp-time_distributed_4/kernel/Read/ReadVariableOp+time_distributed_4/bias/Read/ReadVariableOp-time_distributed_6/kernel/Read/ReadVariableOp+time_distributed_6/bias/Read/ReadVariableOp-time_distributed_7/kernel/Read/ReadVariableOp+time_distributed_7/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_301269
Ù
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biastime_distributed_3/kerneltime_distributed_3/biastime_distributed_4/kerneltime_distributed_4/biastime_distributed_6/kerneltime_distributed_6/biastime_distributed_7/kerneltime_distributed_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_301315÷Å
í#

L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_299765

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ý
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿)
ç
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_300974

inputsU
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity¢)conv2d_transpose_1/BiasAdd/ReadVariableOp¢2conv2d_transpose_1/conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
conv2d_transpose_1/ShapeShapeReshape:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :è
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0À
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape#conv2d_transpose_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
NoOpNoOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾!

H__inference_sequential_1_layer_call_and_return_conditional_losses_300042

inputs,
time_distributed_4_300017:	 @(
time_distributed_4_300019:	@3
time_distributed_6_300027: '
time_distributed_6_300029:3
time_distributed_7_300034:'
time_distributed_7_300036:
identity¢*time_distributed_4/StatefulPartitionedCall¢*time_distributed_6/StatefulPartitionedCall¢*time_distributed_7/StatefulPartitionedCall 
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_300017time_distributed_4_300019*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_299647q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
time_distributed_4/ReshapeReshapeinputs)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_299720q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0time_distributed_6_300027time_distributed_6_300029*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_299829y
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ·
time_distributed_6/ReshapeReshape+time_distributed_5/PartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ô
*time_distributed_7/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0time_distributed_7_300034time_distributed_7_300036*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_299941y
 time_distributed_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_7/ReshapeReshape3time_distributed_6/StatefulPartitionedCall:output:0)time_distributed_7/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity3time_distributed_7/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿÍ
NoOpNoOp+^time_distributed_4/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall+^time_distributed_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall2X
*time_distributed_7/StatefulPartitionedCall*time_distributed_7/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥	

+__inference_sequential_layer_call_fn_299437
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:	@@
	unknown_4:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_299422s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ú
O
3__inference_time_distributed_5_layer_call_fn_300774

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_299720u
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§	

-__inference_sequential_1_layer_call_fn_300298

inputs
unknown:	 @
	unknown_0:	@#
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_300042{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¶
¨
3__inference_time_distributed_6_layer_call_fn_300837

inputs!
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_299798
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

ý
D__inference_conv2d_1_layer_call_and_return_conditional_losses_301050

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¨
3__inference_conv2d_transpose_1_layer_call_fn_301175

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_299877
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

)__inference_conv2d_1_layer_call_fn_301039

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_299191w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
¨
3__inference_time_distributed_7_layer_call_fn_300929

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_299910
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_300828

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
reshape/Shape_1ShapeReshape:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
reshape/strided_sliceStridedSlicereshape/Shape_1:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ñ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshapeReshape:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapereshape/Reshape:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¢
3__inference_time_distributed_4_layer_call_fn_300720

inputs
unknown:	 @
	unknown_0:	@
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_299647}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Ê
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_299383

inputs
dense_299373:	@@
dense_299375:@
identity¢dense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_299373dense_299375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_299333\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾!

H__inference_sequential_1_layer_call_and_return_conditional_losses_299980

inputs,
time_distributed_4_299955:	 @(
time_distributed_4_299957:	@3
time_distributed_6_299965: '
time_distributed_6_299967:3
time_distributed_7_299972:'
time_distributed_7_299974:
identity¢*time_distributed_4/StatefulPartitionedCall¢*time_distributed_6/StatefulPartitionedCall¢*time_distributed_7/StatefulPartitionedCall 
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_4_299955time_distributed_4_299957*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_299608q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
time_distributed_4/ReshapeReshapeinputs)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_299691q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0time_distributed_6_299965time_distributed_6_299967*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_299798y
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ·
time_distributed_6/ReshapeReshape+time_distributed_5/PartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ô
*time_distributed_7/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0time_distributed_7_299972time_distributed_7_299974*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_299910y
 time_distributed_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_7/ReshapeReshape3time_distributed_6/StatefulPartitionedCall:output:0)time_distributed_7/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity3time_distributed_7/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿÍ
NoOpNoOp+^time_distributed_4/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall+^time_distributed_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall2X
*time_distributed_7/StatefulPartitionedCall*time_distributed_7/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
É

N__inference_time_distributed_3_layer_call_and_return_conditional_losses_300702

inputs7
$dense_matmul_readvariableop_resource:	@@3
%dense_biasadd_readvariableop_resource:@
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@@*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â

&__inference_dense_layer_call_fn_301070

inputs
unknown:	@@
	unknown_0:@
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_299333o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
¡
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_300742

inputs9
&dense_1_matmul_readvariableop_resource:	 @6
'dense_1_biasadd_readvariableop_resource:	@
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:@*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢

ö
C__inference_dense_1_layer_call_and_return_conditional_losses_301100

inputs1
matmul_readvariableop_resource:	 @.
biasadd_readvariableop_resource:	@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:@*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ú
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_300642

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¯
Ñ
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_299608

inputs!
dense_1_299598:	 @
dense_1_299600:	@
identity¢dense_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ú
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_1_299598dense_1_299600*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_299597\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
²
¦
1__inference_time_distributed_layer_call_fn_300484

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_299159
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

N__inference_time_distributed_3_layer_call_and_return_conditional_losses_300681

inputs7
$dense_matmul_readvariableop_resource:	@@3
%dense_biasadd_readvariableop_resource:@
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@@*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â!

H__inference_sequential_1_layer_call_and_return_conditional_losses_300130
input_2,
time_distributed_4_300105:	 @(
time_distributed_4_300107:	@3
time_distributed_6_300115: '
time_distributed_6_300117:3
time_distributed_7_300122:'
time_distributed_7_300124:
identity¢*time_distributed_4/StatefulPartitionedCall¢*time_distributed_6/StatefulPartitionedCall¢*time_distributed_7/StatefulPartitionedCall¡
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCallinput_2time_distributed_4_300105time_distributed_4_300107*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_299647q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
time_distributed_4/ReshapeReshapeinput_2)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_299720q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0time_distributed_6_300115time_distributed_6_300117*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_299829y
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ·
time_distributed_6/ReshapeReshape+time_distributed_5/PartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ô
*time_distributed_7/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0time_distributed_7_300122time_distributed_7_300124*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_299941y
 time_distributed_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_7/ReshapeReshape3time_distributed_6/StatefulPartitionedCall:output:0)time_distributed_7/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity3time_distributed_7/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿÍ
NoOpNoOp+^time_distributed_4/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall+^time_distributed_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall2X
*time_distributed_7/StatefulPartitionedCall*time_distributed_7/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_2
É
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_299720

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
reshape/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_299682\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape reshape/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ
¦
1__inference_conv2d_transpose_layer_call_fn_301128

inputs!
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_299765
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

ý
D__inference_conv2d_1_layer_call_and_return_conditional_losses_299191

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
ø
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_299910

inputs3
conv2d_transpose_1_299898:'
conv2d_transpose_1_299900:
identity¢*conv2d_transpose_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_transpose_1_299898conv2d_transpose_1_299900*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_299877\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:ª
	Reshape_1Reshape3conv2d_transpose_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
NoOpNoOp+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í#

L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_301166

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ý
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ú
O
3__inference_time_distributed_2_layer_call_fn_300603

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_299279n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
á)
ß
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_300883

inputsS
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:
identity¢'conv2d_transpose/BiasAdd/ReadVariableOp¢0conv2d_transpose/conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
conv2d_transpose/ShapeShapeReshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Þ
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask²
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0º
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape#conv2d_transpose/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Çz
ë
H__inference_sequential_1_layer_call_and_return_conditional_losses_300382

inputsL
9time_distributed_4_dense_1_matmul_readvariableop_resource:	 @I
:time_distributed_4_dense_1_biasadd_readvariableop_resource:	@f
Ltime_distributed_6_conv2d_transpose_conv2d_transpose_readvariableop_resource: Q
Ctime_distributed_6_conv2d_transpose_biasadd_readvariableop_resource:h
Ntime_distributed_7_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:S
Etime_distributed_7_conv2d_transpose_1_biasadd_readvariableop_resource:
identity¢1time_distributed_4/dense_1/BiasAdd/ReadVariableOp¢0time_distributed_4/dense_1/MatMul/ReadVariableOp¢:time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOp¢Ctime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOp¢<time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOp¢Etime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOpq
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
time_distributed_4/ReshapeReshapeinputs)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ «
0time_distributed_4/dense_1/MatMul/ReadVariableOpReadVariableOp9time_distributed_4_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0½
!time_distributed_4/dense_1/MatMulMatMul#time_distributed_4/Reshape:output:08time_distributed_4/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
1time_distributed_4/dense_1/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_4_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:@*
dtype0È
"time_distributed_4/dense_1/BiasAddBiasAdd+time_distributed_4/dense_1/MatMul:product:09time_distributed_4/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
time_distributed_4/dense_1/ReluRelu+time_distributed_4/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
"time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ       º
time_distributed_4/Reshape_1Reshape-time_distributed_4/dense_1/Relu:activations:0+time_distributed_4/Reshape_1/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
"time_distributed_4/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
time_distributed_4/Reshape_2Reshapeinputs+time_distributed_4/Reshape_2/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ª
time_distributed_5/ReshapeReshape%time_distributed_4/Reshape_1:output:0)time_distributed_5/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
"time_distributed_5/reshape/Shape_1Shape#time_distributed_5/Reshape:output:0*
T0*
_output_shapes
:x
.time_distributed_5/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0time_distributed_5/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0time_distributed_5/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ú
(time_distributed_5/reshape/strided_sliceStridedSlice+time_distributed_5/reshape/Shape_1:output:07time_distributed_5/reshape/strided_slice/stack:output:09time_distributed_5/reshape/strided_slice/stack_1:output:09time_distributed_5/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*time_distributed_5/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :l
*time_distributed_5/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :l
*time_distributed_5/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : °
(time_distributed_5/reshape/Reshape/shapePack1time_distributed_5/reshape/strided_slice:output:03time_distributed_5/reshape/Reshape/shape/1:output:03time_distributed_5/reshape/Reshape/shape/2:output:03time_distributed_5/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¿
"time_distributed_5/reshape/ReshapeReshape#time_distributed_5/Reshape:output:01time_distributed_5/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ             ¿
time_distributed_5/Reshape_1Reshape+time_distributed_5/reshape/Reshape:output:0+time_distributed_5/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ s
"time_distributed_5/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ®
time_distributed_5/Reshape_2Reshape%time_distributed_4/Reshape_1:output:0+time_distributed_5/Reshape_2/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ±
time_distributed_6/ReshapeReshape%time_distributed_5/Reshape_1:output:0)time_distributed_6/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
)time_distributed_6/conv2d_transpose/ShapeShape#time_distributed_6/Reshape:output:0*
T0*
_output_shapes
:
7time_distributed_6/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9time_distributed_6/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9time_distributed_6/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1time_distributed_6/conv2d_transpose/strided_sliceStridedSlice2time_distributed_6/conv2d_transpose/Shape:output:0@time_distributed_6/conv2d_transpose/strided_slice/stack:output:0Btime_distributed_6/conv2d_transpose/strided_slice/stack_1:output:0Btime_distributed_6/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+time_distributed_6/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m
+time_distributed_6/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :m
+time_distributed_6/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :½
)time_distributed_6/conv2d_transpose/stackPack:time_distributed_6/conv2d_transpose/strided_slice:output:04time_distributed_6/conv2d_transpose/stack/1:output:04time_distributed_6/conv2d_transpose/stack/2:output:04time_distributed_6/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:
9time_distributed_6/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;time_distributed_6/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;time_distributed_6/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3time_distributed_6/conv2d_transpose/strided_slice_1StridedSlice2time_distributed_6/conv2d_transpose/stack:output:0Btime_distributed_6/conv2d_transpose/strided_slice_1/stack:output:0Dtime_distributed_6/conv2d_transpose/strided_slice_1/stack_1:output:0Dtime_distributed_6/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskØ
Ctime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpLtime_distributed_6_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ô
4time_distributed_6/conv2d_transpose/conv2d_transposeConv2DBackpropInput2time_distributed_6/conv2d_transpose/stack:output:0Ktime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0#time_distributed_6/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
º
:time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpCtime_distributed_6_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ó
+time_distributed_6/conv2d_transpose/BiasAddBiasAdd=time_distributed_6/conv2d_transpose/conv2d_transpose:output:0Btime_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(time_distributed_6/conv2d_transpose/ReluRelu4time_distributed_6/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"time_distributed_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ            Ê
time_distributed_6/Reshape_1Reshape6time_distributed_6/conv2d_transpose/Relu:activations:0+time_distributed_6/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_6/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          µ
time_distributed_6/Reshape_2Reshape%time_distributed_5/Reshape_1:output:0+time_distributed_6/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
 time_distributed_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ±
time_distributed_7/ReshapeReshape%time_distributed_6/Reshape_1:output:0)time_distributed_7/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
+time_distributed_7/conv2d_transpose_1/ShapeShape#time_distributed_7/Reshape:output:0*
T0*
_output_shapes
:
9time_distributed_7/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;time_distributed_7/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;time_distributed_7/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3time_distributed_7/conv2d_transpose_1/strided_sliceStridedSlice4time_distributed_7/conv2d_transpose_1/Shape:output:0Btime_distributed_7/conv2d_transpose_1/strided_slice/stack:output:0Dtime_distributed_7/conv2d_transpose_1/strided_slice/stack_1:output:0Dtime_distributed_7/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-time_distributed_7/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :o
-time_distributed_7/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :o
-time_distributed_7/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ç
+time_distributed_7/conv2d_transpose_1/stackPack<time_distributed_7/conv2d_transpose_1/strided_slice:output:06time_distributed_7/conv2d_transpose_1/stack/1:output:06time_distributed_7/conv2d_transpose_1/stack/2:output:06time_distributed_7/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:
;time_distributed_7/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=time_distributed_7/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=time_distributed_7/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5time_distributed_7/conv2d_transpose_1/strided_slice_1StridedSlice4time_distributed_7/conv2d_transpose_1/stack:output:0Dtime_distributed_7/conv2d_transpose_1/strided_slice_1/stack:output:0Ftime_distributed_7/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ftime_distributed_7/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÜ
Etime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpNtime_distributed_7_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ú
6time_distributed_7/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput4time_distributed_7/conv2d_transpose_1/stack:output:0Mtime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#time_distributed_7/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
¾
<time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpEtime_distributed_7_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ù
-time_distributed_7/conv2d_transpose_1/BiasAddBiasAdd?time_distributed_7/conv2d_transpose_1/conv2d_transpose:output:0Dtime_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"time_distributed_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ            Ê
time_distributed_7/Reshape_1Reshape6time_distributed_7/conv2d_transpose_1/BiasAdd:output:0+time_distributed_7/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_7/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         µ
time_distributed_7/Reshape_2Reshape%time_distributed_6/Reshape_1:output:0+time_distributed_7/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity%time_distributed_7/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp2^time_distributed_4/dense_1/BiasAdd/ReadVariableOp1^time_distributed_4/dense_1/MatMul/ReadVariableOp;^time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOpD^time_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOp=^time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOpF^time_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2f
1time_distributed_4/dense_1/BiasAdd/ReadVariableOp1time_distributed_4/dense_1/BiasAdd/ReadVariableOp2d
0time_distributed_4/dense_1/MatMul/ReadVariableOp0time_distributed_4/dense_1/MatMul/ReadVariableOp2x
:time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOp:time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOp2
Ctime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOpCtime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOp2|
<time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOp<time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOp2
Etime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOpEtime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¡
L__inference_time_distributed_layer_call_and_return_conditional_losses_300532

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0²
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

û
B__inference_conv2d_layer_call_and_return_conditional_losses_299105

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
ø
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_299941

inputs3
conv2d_transpose_1_299929:'
conv2d_transpose_1_299931:
identity¢*conv2d_transpose_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_transpose_1_299929conv2d_transpose_1_299931*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_299877\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:ª
	Reshape_1Reshape3conv2d_transpose_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
NoOpNoOp+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
«
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_300574

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: 
identity¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¶
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§	

-__inference_sequential_1_layer_call_fn_300281

inputs
unknown:	 @
	unknown_0:	@#
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_299980{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ê
«
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_300598

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: 
identity¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¶
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±!
ú
F__inference_sequential_layer_call_and_return_conditional_losses_299544
input_11
time_distributed_299519:%
time_distributed_299521:3
time_distributed_1_299526: '
time_distributed_1_299528: ,
time_distributed_3_299536:	@@'
time_distributed_3_299538:@
identity¢(time_distributed/StatefulPartitionedCall¢*time_distributed_1/StatefulPartitionedCall¢*time_distributed_3/StatefulPartitionedCall 
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinput_1time_distributed_299519time_distributed_299521*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_299118w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
time_distributed/ReshapeReshapeinput_1'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_299526time_distributed_1_299528*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_299204y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ½
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_299279y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¿
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
*time_distributed_3/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_2/PartitionedCall:output:0time_distributed_3_299536time_distributed_3_299538*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_299344q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    °
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
IdentityIdentity3time_distributed_3/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
NoOpNoOp)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_3/StatefulPartitionedCall*time_distributed_3/StatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¶
¨
3__inference_time_distributed_6_layer_call_fn_300846

inputs!
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_299829
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±!
ú
F__inference_sequential_layer_call_and_return_conditional_losses_299572
input_11
time_distributed_299547:%
time_distributed_299549:3
time_distributed_1_299554: '
time_distributed_1_299556: ,
time_distributed_3_299564:	@@'
time_distributed_3_299566:@
identity¢(time_distributed/StatefulPartitionedCall¢*time_distributed_1/StatefulPartitionedCall¢*time_distributed_3/StatefulPartitionedCall 
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinput_1time_distributed_299547time_distributed_299549*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_299159w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
time_distributed/ReshapeReshapeinput_1'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_299554time_distributed_1_299556*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_299245y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ½
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_299306y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¿
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
*time_distributed_3/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_2/PartitionedCall:output:0time_distributed_3_299564time_distributed_3_299566*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_299383q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    °
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
IdentityIdentity3time_distributed_3/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
NoOpNoOp)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_3/StatefulPartitionedCall*time_distributed_3/StatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¢	

+__inference_sequential_layer_call_fn_300164

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:	@@
	unknown_4:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_299484s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Ò
L__inference_time_distributed_layer_call_and_return_conditional_losses_299118

inputs'
conv2d_299106:
conv2d_299108:
identity¢conv2d/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_299106conv2d_299108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_299105\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_299306

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_299272\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢	

+__inference_sequential_layer_call_fn_300147

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:	@@
	unknown_4:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_299422s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª	

-__inference_sequential_1_layer_call_fn_300074
input_2
unknown:	 @
	unknown_0:	@#
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_300042{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_2
àC
à
F__inference_sequential_layer_call_and_return_conditional_losses_300264

inputsP
6time_distributed_conv2d_conv2d_readvariableop_resource:E
7time_distributed_conv2d_biasadd_readvariableop_resource:T
:time_distributed_1_conv2d_1_conv2d_readvariableop_resource: I
;time_distributed_1_conv2d_1_biasadd_readvariableop_resource: J
7time_distributed_3_dense_matmul_readvariableop_resource:	@@F
8time_distributed_3_dense_biasadd_readvariableop_resource:@
identity¢.time_distributed/conv2d/BiasAdd/ReadVariableOp¢-time_distributed/conv2d/Conv2D/ReadVariableOp¢2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp¢1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp¢/time_distributed_3/dense/BiasAdd/ReadVariableOp¢.time_distributed_3/dense/MatMul/ReadVariableOpw
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0å
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
¢
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ            º
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿy
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¯
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
1time_distributed_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:time_distributed_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ï
"time_distributed_1/conv2d_1/Conv2DConv2D#time_distributed_1/Reshape:output:09time_distributed_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
ª
2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;time_distributed_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ñ
#time_distributed_1/conv2d_1/BiasAddBiasAdd+time_distributed_1/conv2d_1/Conv2D:output:0:time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 time_distributed_1/conv2d_1/ReluRelu,time_distributed_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ             Â
time_distributed_1/Reshape_1Reshape.time_distributed_1/conv2d_1/Relu:activations:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ {
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ³
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ±
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
 time_distributed_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    °
"time_distributed_2/flatten/ReshapeReshape#time_distributed_2/Reshape:output:0)time_distributed_2/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ       ¸
time_distributed_2/Reshape_1Reshape+time_distributed_2/flatten/Reshape:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          µ
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ª
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
.time_distributed_3/dense/MatMul/ReadVariableOpReadVariableOp7time_distributed_3_dense_matmul_readvariableop_resource*
_output_shapes
:	@@*
dtype0¸
time_distributed_3/dense/MatMulMatMul#time_distributed_3/Reshape:output:06time_distributed_3/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
/time_distributed_3/dense/BiasAdd/ReadVariableOpReadVariableOp8time_distributed_3_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Á
 time_distributed_3/dense/BiasAddBiasAdd)time_distributed_3/dense/MatMul:product:07time_distributed_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   µ
time_distributed_3/Reshape_1Reshape)time_distributed_3/dense/BiasAdd:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ®
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
IdentityIdentity%time_distributed_3/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ó
NoOpNoOp/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp3^time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2^time_distributed_1/conv2d_1/Conv2D/ReadVariableOp0^time_distributed_3/dense/BiasAdd/ReadVariableOp/^time_distributed_3/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp2h
2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2f
1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp2b
/time_distributed_3/dense/BiasAdd/ReadVariableOp/time_distributed_3/dense/BiasAdd/ReadVariableOp2`
.time_distributed_3/dense/MatMul/ReadVariableOp.time_distributed_3/dense/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
D
(__inference_reshape_layer_call_fn_301105

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_299682h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç

(__inference_dense_1_layer_call_fn_301089

inputs
unknown:	 @
	unknown_0:	@
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_299597p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ì
_
C__inference_reshape_layer_call_and_return_conditional_losses_301119

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
àC
à
F__inference_sequential_layer_call_and_return_conditional_losses_300214

inputsP
6time_distributed_conv2d_conv2d_readvariableop_resource:E
7time_distributed_conv2d_biasadd_readvariableop_resource:T
:time_distributed_1_conv2d_1_conv2d_readvariableop_resource: I
;time_distributed_1_conv2d_1_biasadd_readvariableop_resource: J
7time_distributed_3_dense_matmul_readvariableop_resource:	@@F
8time_distributed_3_dense_biasadd_readvariableop_resource:@
identity¢.time_distributed/conv2d/BiasAdd/ReadVariableOp¢-time_distributed/conv2d/Conv2D/ReadVariableOp¢2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp¢1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp¢/time_distributed_3/dense/BiasAdd/ReadVariableOp¢.time_distributed_3/dense/MatMul/ReadVariableOpw
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0å
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
¢
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ            º
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿy
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¯
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
1time_distributed_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:time_distributed_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ï
"time_distributed_1/conv2d_1/Conv2DConv2D#time_distributed_1/Reshape:output:09time_distributed_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
ª
2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;time_distributed_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ñ
#time_distributed_1/conv2d_1/BiasAddBiasAdd+time_distributed_1/conv2d_1/Conv2D:output:0:time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 time_distributed_1/conv2d_1/ReluRelu,time_distributed_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ             Â
time_distributed_1/Reshape_1Reshape.time_distributed_1/conv2d_1/Relu:activations:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ {
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ³
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ±
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
 time_distributed_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    °
"time_distributed_2/flatten/ReshapeReshape#time_distributed_2/Reshape:output:0)time_distributed_2/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ       ¸
time_distributed_2/Reshape_1Reshape+time_distributed_2/flatten/Reshape:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          µ
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ª
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@§
.time_distributed_3/dense/MatMul/ReadVariableOpReadVariableOp7time_distributed_3_dense_matmul_readvariableop_resource*
_output_shapes
:	@@*
dtype0¸
time_distributed_3/dense/MatMulMatMul#time_distributed_3/Reshape:output:06time_distributed_3/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
/time_distributed_3/dense/BiasAdd/ReadVariableOpReadVariableOp8time_distributed_3_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Á
 time_distributed_3/dense/BiasAddBiasAdd)time_distributed_3/dense/MatMul:product:07time_distributed_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ   @   µ
time_distributed_3/Reshape_1Reshape)time_distributed_3/dense/BiasAdd:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ®
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
IdentityIdentity%time_distributed_3/Reshape_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ó
NoOpNoOp/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp3^time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2^time_distributed_1/conv2d_1/Conv2D/ReadVariableOp0^time_distributed_3/dense/BiasAdd/ReadVariableOp/^time_distributed_3/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp2h
2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2f
1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp2b
/time_distributed_3/dense/BiasAdd/ReadVariableOp/time_distributed_3/dense/BiasAdd/ReadVariableOp2`
.time_distributed_3/dense/MatMul/ReadVariableOp.time_distributed_3/dense/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
Ú
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_299204

inputs)
conv2d_1_299192: 
conv2d_1_299194: 
identity¢ conv2d_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_1_299192conv2d_1_299194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_299191\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ i
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥	

+__inference_sequential_layer_call_fn_299516
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:	@@
	unknown_4:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_299484s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

¡
3__inference_time_distributed_3_layer_call_fn_300660

inputs
unknown:	@@
	unknown_0:@
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_299383|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
#

N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_301212

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ý
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
Ñ
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_299647

inputs!
dense_1_299637:	 @
dense_1_299639:	@
identity¢dense_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ú
dense_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_1_299637dense_1_299639*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_299597\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape(dense_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
NoOpNoOp ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢

ö
C__inference_dense_1_layer_call_and_return_conditional_losses_299597

inputs1
matmul_readvariableop_resource:	 @.
biasadd_readvariableop_resource:	@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:@*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Çz
ë
H__inference_sequential_1_layer_call_and_return_conditional_losses_300466

inputsL
9time_distributed_4_dense_1_matmul_readvariableop_resource:	 @I
:time_distributed_4_dense_1_biasadd_readvariableop_resource:	@f
Ltime_distributed_6_conv2d_transpose_conv2d_transpose_readvariableop_resource: Q
Ctime_distributed_6_conv2d_transpose_biasadd_readvariableop_resource:h
Ntime_distributed_7_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:S
Etime_distributed_7_conv2d_transpose_1_biasadd_readvariableop_resource:
identity¢1time_distributed_4/dense_1/BiasAdd/ReadVariableOp¢0time_distributed_4/dense_1/MatMul/ReadVariableOp¢:time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOp¢Ctime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOp¢<time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOp¢Etime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOpq
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
time_distributed_4/ReshapeReshapeinputs)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ «
0time_distributed_4/dense_1/MatMul/ReadVariableOpReadVariableOp9time_distributed_4_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0½
!time_distributed_4/dense_1/MatMulMatMul#time_distributed_4/Reshape:output:08time_distributed_4/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
1time_distributed_4/dense_1/BiasAdd/ReadVariableOpReadVariableOp:time_distributed_4_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:@*
dtype0È
"time_distributed_4/dense_1/BiasAddBiasAdd+time_distributed_4/dense_1/MatMul:product:09time_distributed_4/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
time_distributed_4/dense_1/ReluRelu+time_distributed_4/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
"time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ       º
time_distributed_4/Reshape_1Reshape-time_distributed_4/dense_1/Relu:activations:0+time_distributed_4/Reshape_1/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
"time_distributed_4/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
time_distributed_4/Reshape_2Reshapeinputs+time_distributed_4/Reshape_2/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ª
time_distributed_5/ReshapeReshape%time_distributed_4/Reshape_1:output:0)time_distributed_5/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
"time_distributed_5/reshape/Shape_1Shape#time_distributed_5/Reshape:output:0*
T0*
_output_shapes
:x
.time_distributed_5/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0time_distributed_5/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0time_distributed_5/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ú
(time_distributed_5/reshape/strided_sliceStridedSlice+time_distributed_5/reshape/Shape_1:output:07time_distributed_5/reshape/strided_slice/stack:output:09time_distributed_5/reshape/strided_slice/stack_1:output:09time_distributed_5/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*time_distributed_5/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :l
*time_distributed_5/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :l
*time_distributed_5/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : °
(time_distributed_5/reshape/Reshape/shapePack1time_distributed_5/reshape/strided_slice:output:03time_distributed_5/reshape/Reshape/shape/1:output:03time_distributed_5/reshape/Reshape/shape/2:output:03time_distributed_5/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¿
"time_distributed_5/reshape/ReshapeReshape#time_distributed_5/Reshape:output:01time_distributed_5/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ             ¿
time_distributed_5/Reshape_1Reshape+time_distributed_5/reshape/Reshape:output:0+time_distributed_5/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ s
"time_distributed_5/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ®
time_distributed_5/Reshape_2Reshape%time_distributed_4/Reshape_1:output:0+time_distributed_5/Reshape_2/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ±
time_distributed_6/ReshapeReshape%time_distributed_5/Reshape_1:output:0)time_distributed_6/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
)time_distributed_6/conv2d_transpose/ShapeShape#time_distributed_6/Reshape:output:0*
T0*
_output_shapes
:
7time_distributed_6/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9time_distributed_6/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9time_distributed_6/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1time_distributed_6/conv2d_transpose/strided_sliceStridedSlice2time_distributed_6/conv2d_transpose/Shape:output:0@time_distributed_6/conv2d_transpose/strided_slice/stack:output:0Btime_distributed_6/conv2d_transpose/strided_slice/stack_1:output:0Btime_distributed_6/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+time_distributed_6/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m
+time_distributed_6/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :m
+time_distributed_6/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :½
)time_distributed_6/conv2d_transpose/stackPack:time_distributed_6/conv2d_transpose/strided_slice:output:04time_distributed_6/conv2d_transpose/stack/1:output:04time_distributed_6/conv2d_transpose/stack/2:output:04time_distributed_6/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:
9time_distributed_6/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;time_distributed_6/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;time_distributed_6/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3time_distributed_6/conv2d_transpose/strided_slice_1StridedSlice2time_distributed_6/conv2d_transpose/stack:output:0Btime_distributed_6/conv2d_transpose/strided_slice_1/stack:output:0Dtime_distributed_6/conv2d_transpose/strided_slice_1/stack_1:output:0Dtime_distributed_6/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskØ
Ctime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpLtime_distributed_6_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ô
4time_distributed_6/conv2d_transpose/conv2d_transposeConv2DBackpropInput2time_distributed_6/conv2d_transpose/stack:output:0Ktime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0#time_distributed_6/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
º
:time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpCtime_distributed_6_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ó
+time_distributed_6/conv2d_transpose/BiasAddBiasAdd=time_distributed_6/conv2d_transpose/conv2d_transpose:output:0Btime_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(time_distributed_6/conv2d_transpose/ReluRelu4time_distributed_6/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"time_distributed_6/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ            Ê
time_distributed_6/Reshape_1Reshape6time_distributed_6/conv2d_transpose/Relu:activations:0+time_distributed_6/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_6/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          µ
time_distributed_6/Reshape_2Reshape%time_distributed_5/Reshape_1:output:0+time_distributed_6/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ y
 time_distributed_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ±
time_distributed_7/ReshapeReshape%time_distributed_6/Reshape_1:output:0)time_distributed_7/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
+time_distributed_7/conv2d_transpose_1/ShapeShape#time_distributed_7/Reshape:output:0*
T0*
_output_shapes
:
9time_distributed_7/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;time_distributed_7/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;time_distributed_7/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3time_distributed_7/conv2d_transpose_1/strided_sliceStridedSlice4time_distributed_7/conv2d_transpose_1/Shape:output:0Btime_distributed_7/conv2d_transpose_1/strided_slice/stack:output:0Dtime_distributed_7/conv2d_transpose_1/strided_slice/stack_1:output:0Dtime_distributed_7/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-time_distributed_7/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :o
-time_distributed_7/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :o
-time_distributed_7/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ç
+time_distributed_7/conv2d_transpose_1/stackPack<time_distributed_7/conv2d_transpose_1/strided_slice:output:06time_distributed_7/conv2d_transpose_1/stack/1:output:06time_distributed_7/conv2d_transpose_1/stack/2:output:06time_distributed_7/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:
;time_distributed_7/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=time_distributed_7/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=time_distributed_7/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5time_distributed_7/conv2d_transpose_1/strided_slice_1StridedSlice4time_distributed_7/conv2d_transpose_1/stack:output:0Dtime_distributed_7/conv2d_transpose_1/strided_slice_1/stack:output:0Ftime_distributed_7/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ftime_distributed_7/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÜ
Etime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpNtime_distributed_7_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ú
6time_distributed_7/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput4time_distributed_7/conv2d_transpose_1/stack:output:0Mtime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#time_distributed_7/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
¾
<time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpEtime_distributed_7_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ù
-time_distributed_7/conv2d_transpose_1/BiasAddBiasAdd?time_distributed_7/conv2d_transpose_1/conv2d_transpose:output:0Dtime_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"time_distributed_7/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ            Ê
time_distributed_7/Reshape_1Reshape6time_distributed_7/conv2d_transpose_1/BiasAdd:output:0+time_distributed_7/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ{
"time_distributed_7/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         µ
time_distributed_7/Reshape_2Reshape%time_distributed_6/Reshape_1:output:0+time_distributed_7/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity%time_distributed_7/Reshape_1:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp2^time_distributed_4/dense_1/BiasAdd/ReadVariableOp1^time_distributed_4/dense_1/MatMul/ReadVariableOp;^time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOpD^time_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOp=^time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOpF^time_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2f
1time_distributed_4/dense_1/BiasAdd/ReadVariableOp1time_distributed_4/dense_1/BiasAdd/ReadVariableOp2d
0time_distributed_4/dense_1/MatMul/ReadVariableOp0time_distributed_4/dense_1/MatMul/ReadVariableOp2x
:time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOp:time_distributed_6/conv2d_transpose/BiasAdd/ReadVariableOp2
Ctime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOpCtime_distributed_6/conv2d_transpose/conv2d_transpose/ReadVariableOp2|
<time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOp<time_distributed_7/conv2d_transpose_1/BiasAdd/ReadVariableOp2
Etime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOpEtime_distributed_7/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°
D
(__inference_flatten_layer_call_fn_301055

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_299272a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­!
ù
F__inference_sequential_layer_call_and_return_conditional_losses_299484

inputs1
time_distributed_299459:%
time_distributed_299461:3
time_distributed_1_299466: '
time_distributed_1_299468: ,
time_distributed_3_299476:	@@'
time_distributed_3_299478:@
identity¢(time_distributed/StatefulPartitionedCall¢*time_distributed_1/StatefulPartitionedCall¢*time_distributed_3/StatefulPartitionedCall
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_299459time_distributed_299461*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_299159w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_299466time_distributed_1_299468*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_299245y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ½
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_299306y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¿
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
*time_distributed_3/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_2/PartitionedCall:output:0time_distributed_3_299476time_distributed_3_299478*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_299383q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    °
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
IdentityIdentity3time_distributed_3/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
NoOpNoOp)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_3/StatefulPartitionedCall*time_distributed_3/StatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â!

H__inference_sequential_1_layer_call_and_return_conditional_losses_300102
input_2,
time_distributed_4_300077:	 @(
time_distributed_4_300079:	@3
time_distributed_6_300087: '
time_distributed_6_300089:3
time_distributed_7_300094:'
time_distributed_7_300096:
identity¢*time_distributed_4/StatefulPartitionedCall¢*time_distributed_6/StatefulPartitionedCall¢*time_distributed_7/StatefulPartitionedCall¡
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCallinput_2time_distributed_4_300077time_distributed_4_300079*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_299608q
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
time_distributed_4/ReshapeReshapeinput_2)time_distributed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_299691q
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
time_distributed_5/ReshapeReshape3time_distributed_4/StatefulPartitionedCall:output:0)time_distributed_5/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ì
*time_distributed_6/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0time_distributed_6_300087time_distributed_6_300089*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_299798y
 time_distributed_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ·
time_distributed_6/ReshapeReshape+time_distributed_5/PartitionedCall:output:0)time_distributed_6/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ô
*time_distributed_7/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_6/StatefulPartitionedCall:output:0time_distributed_7_300094time_distributed_7_300096*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_299910y
 time_distributed_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ¿
time_distributed_7/ReshapeReshape3time_distributed_6/StatefulPartitionedCall:output:0)time_distributed_7/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity3time_distributed_7/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿÍ
NoOpNoOp+^time_distributed_4/StatefulPartitionedCall+^time_distributed_6/StatefulPartitionedCall+^time_distributed_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall2X
*time_distributed_6/StatefulPartitionedCall*time_distributed_6/StatefulPartitionedCall2X
*time_distributed_7/StatefulPartitionedCall*time_distributed_7/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_2

Ê
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_299344

inputs
dense_299334:	@@
dense_299336:@
identity¢dense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ñ
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_299334dense_299336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_299333\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@n
IdentityIdentityReshape_1:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¡
3__inference_time_distributed_3_layer_call_fn_300651

inputs
unknown:	@@
	unknown_0:@
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_299344|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ö
Ú
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_299245

inputs)
conv2d_1_299233: 
conv2d_1_299235: 
identity¢ conv2d_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_1_299233conv2d_1_299235*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_299191\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
: 
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ i
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_300625

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ð
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_299279

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_299272\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ð$

__inference__traced_save_301269
file_prefix6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop8
4savev2_time_distributed_1_kernel_read_readvariableop6
2savev2_time_distributed_1_bias_read_readvariableop8
4savev2_time_distributed_3_kernel_read_readvariableop6
2savev2_time_distributed_3_bias_read_readvariableop8
4savev2_time_distributed_4_kernel_read_readvariableop6
2savev2_time_distributed_4_bias_read_readvariableop8
4savev2_time_distributed_6_kernel_read_readvariableop6
2savev2_time_distributed_6_bias_read_readvariableop8
4savev2_time_distributed_7_kernel_read_readvariableop6
2savev2_time_distributed_7_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ê
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó
valueéBæB.encoder/variables/0/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/1/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/2/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/3/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/4/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/5/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/0/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/1/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/2/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/3/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/4/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ´
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_1_kernel_read_readvariableop2savev2_time_distributed_1_bias_read_readvariableop4savev2_time_distributed_3_kernel_read_readvariableop2savev2_time_distributed_3_bias_read_readvariableop4savev2_time_distributed_4_kernel_read_readvariableop2savev2_time_distributed_4_bias_read_readvariableop4savev2_time_distributed_6_kernel_read_readvariableop2savev2_time_distributed_6_bias_read_readvariableop4savev2_time_distributed_7_kernel_read_readvariableop2savev2_time_distributed_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: ::: : :	@@:@:	 @:@: :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	@@: 

_output_shapes
:@:%!

_output_shapes
:	 @:!

_output_shapes	
:@:,	(
&
_output_shapes
: : 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
é

'__inference_conv2d_layer_call_fn_301019

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_299105w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿)
ç
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_301010

inputsU
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:
identity¢)conv2d_transpose_1/BiasAdd/ReadVariableOp¢2conv2d_transpose_1/conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
conv2d_transpose_1/ShapeShapeReshape:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :è
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0À
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape#conv2d_transpose_1/BiasAdd:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
NoOpNoOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
O
3__inference_time_distributed_2_layer_call_fn_300608

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_299306n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¶
¨
3__inference_time_distributed_7_layer_call_fn_300938

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_299941
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á)
ß
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_300920

inputsS
9conv2d_transpose_conv2d_transpose_readvariableop_resource: >
0conv2d_transpose_biasadd_readvariableop_resource:
identity¢'conv2d_transpose/BiasAdd/ReadVariableOp¢0conv2d_transpose/conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ V
conv2d_transpose/ShapeShapeReshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Þ
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask²
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0º
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape#conv2d_transpose/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
é
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_300801

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
reshape/Shape_1ShapeReshape:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
reshape/strided_sliceStridedSlicereshape/Shape_1:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ñ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshapeReshape:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapereshape/Reshape:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

û
B__inference_conv2d_layer_call_and_return_conditional_losses_301030

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¡
L__inference_time_distributed_layer_call_and_return_conditional_losses_300508

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0²
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Ò
L__inference_time_distributed_layer_call_and_return_conditional_losses_299159

inputs'
conv2d_299147:
conv2d_299149:
identity¢conv2d/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_299147conv2d_299149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_299105\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿg
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
ò
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_299829

inputs1
conv2d_transpose_299817: %
conv2d_transpose_299819:
identity¢(conv2d_transpose/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_transpose_299817conv2d_transpose_299819*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_299765\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:¨
	Reshape_1Reshape1conv2d_transpose/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
É
j
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_299691

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
reshape/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_299682\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape reshape/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
²
¦
1__inference_time_distributed_layer_call_fn_300475

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_299118
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
4
À
"__inference__traced_restore_301315
file_prefixB
(assignvariableop_time_distributed_kernel:6
(assignvariableop_1_time_distributed_bias:F
,assignvariableop_2_time_distributed_1_kernel: 8
*assignvariableop_3_time_distributed_1_bias: ?
,assignvariableop_4_time_distributed_3_kernel:	@@8
*assignvariableop_5_time_distributed_3_bias:@?
,assignvariableop_6_time_distributed_4_kernel:	 @9
*assignvariableop_7_time_distributed_4_bias:	@F
,assignvariableop_8_time_distributed_6_kernel: 8
*assignvariableop_9_time_distributed_6_bias:G
-assignvariableop_10_time_distributed_7_kernel:9
+assignvariableop_11_time_distributed_7_bias:
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Í
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó
valueéBæB.encoder/variables/0/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/1/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/2/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/3/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/4/.ATTRIBUTES/VARIABLE_VALUEB.encoder/variables/5/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/0/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/1/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/2/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/3/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/4/.ATTRIBUTES/VARIABLE_VALUEB.decoder/variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ß
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_time_distributed_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp(assignvariableop_1_time_distributed_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp,assignvariableop_2_time_distributed_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp*assignvariableop_3_time_distributed_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp,assignvariableop_4_time_distributed_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp*assignvariableop_5_time_distributed_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_time_distributed_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp*assignvariableop_7_time_distributed_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp,assignvariableop_8_time_distributed_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp*assignvariableop_9_time_distributed_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp-assignvariableop_10_time_distributed_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp+assignvariableop_11_time_distributed_7_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ×
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
È	
ó
A__inference_dense_layer_call_and_return_conditional_losses_301080

inputs1
matmul_readvariableop_resource:	@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®
ò
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_299798

inputs1
conv2d_transpose_299786: %
conv2d_transpose_299788:
identity¢(conv2d_transpose/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_transpose_299786conv2d_transpose_299788*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_299765\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿS
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Í
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:¨
	Reshape_1Reshape1conv2d_transpose/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª	

-__inference_sequential_1_layer_call_fn_299995
input_2
unknown:	 @
	unknown_0:	@#
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_299980{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
!
_user_specified_name	input_2
­!
ù
F__inference_sequential_layer_call_and_return_conditional_losses_299422

inputs1
time_distributed_299397:%
time_distributed_299399:3
time_distributed_1_299404: '
time_distributed_1_299406: ,
time_distributed_3_299414:	@@'
time_distributed_3_299416:@
identity¢(time_distributed/StatefulPartitionedCall¢*time_distributed_1/StatefulPartitionedCall¢*time_distributed_3/StatefulPartitionedCall
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_299397time_distributed_299399*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_299118w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_299404time_distributed_1_299406*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_299204y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         ½
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_299279y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ          ¿
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
*time_distributed_3/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_2/PartitionedCall:output:0time_distributed_3_299414time_distributed_3_299416*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_299344q
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    °
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
IdentityIdentity3time_distributed_3/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ë
NoOpNoOp)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall+^time_distributed_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : 2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall2X
*time_distributed_3/StatefulPartitionedCall*time_distributed_3/StatefulPartitionedCall:[ W
3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
#

N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_299877

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ý
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
O
3__inference_time_distributed_5_layer_call_fn_300769

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_299691u
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¶
¨
3__inference_time_distributed_1_layer_call_fn_300541

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_299204
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
_
C__inference_flatten_layer_call_and_return_conditional_losses_301061

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
¡
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_300764

inputs9
&dense_1_matmul_readvariableop_resource:	 @6
'dense_1_biasadd_readvariableop_resource:	@
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	 @*
dtype0
dense_1/MatMulMatMulReshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:@*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :@
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense_1/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È	
ó
A__inference_dense_layer_call_and_return_conditional_losses_299333

inputs1
matmul_readvariableop_resource:	@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¢
3__inference_time_distributed_4_layer_call_fn_300711

inputs
unknown:	 @
	unknown_0:	@
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_299608}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Å
_
C__inference_flatten_layer_call_and_return_conditional_losses_299272

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ì
_
C__inference_reshape_layer_call_and_return_conditional_losses_299682

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¶
¨
3__inference_time_distributed_1_layer_call_fn_300550

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_299245
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"µ	J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ÞÃ
^
	keras_api
encoder
decoder

sample

signatures"
_tf_keras_model
"
_generic_user_object
¬
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	layer_with_weights-2
	layer-3

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
¬
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
ª2§¤
²
FullArgSpec
args
jself
jeps
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
signature_map
°
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	 layer"
_tf_keras_layer
°
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
	'layer"
_tf_keras_layer
°
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
	.layer"
_tf_keras_layer
°
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
	5layer"
_tf_keras_layer
J
60
71
82
93
:4
;5"
trackable_list_wrapper
J
60
71
82
93
:4
;5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics

	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
á
Atrace_0
Btrace_1
Ctrace_2
Dtrace_32ö
+__inference_sequential_layer_call_fn_299437
+__inference_sequential_layer_call_fn_300147
+__inference_sequential_layer_call_fn_300164
+__inference_sequential_layer_call_fn_299516¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zAtrace_0zBtrace_1zCtrace_2zDtrace_3
Í
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32â
F__inference_sequential_layer_call_and_return_conditional_losses_300214
F__inference_sequential_layer_call_and_return_conditional_losses_300264
F__inference_sequential_layer_call_and_return_conditional_losses_299544
F__inference_sequential_layer_call_and_return_conditional_losses_299572¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
°
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
	Olayer"
_tf_keras_layer
°
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
	Vlayer"
_tf_keras_layer
°
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
	]layer"
_tf_keras_layer
°
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
	dlayer"
_tf_keras_layer
J
e0
f1
g2
h3
i4
j5"
trackable_list_wrapper
J
e0
f1
g2
h3
i4
j5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
é
ptrace_0
qtrace_1
rtrace_2
strace_32þ
-__inference_sequential_1_layer_call_fn_299995
-__inference_sequential_1_layer_call_fn_300281
-__inference_sequential_1_layer_call_fn_300298
-__inference_sequential_1_layer_call_fn_300074¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zptrace_0zqtrace_1zrtrace_2zstrace_3
Õ
ttrace_0
utrace_1
vtrace_2
wtrace_32ê
H__inference_sequential_1_layer_call_and_return_conditional_losses_300382
H__inference_sequential_1_layer_call_and_return_conditional_losses_300466
H__inference_sequential_1_layer_call_and_return_conditional_losses_300102
H__inference_sequential_1_layer_call_and_return_conditional_losses_300130¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zttrace_0zutrace_1zvtrace_2zwtrace_3
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
­
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ß
}trace_0
~trace_12¨
1__inference_time_distributed_layer_call_fn_300475
1__inference_time_distributed_layer_call_fn_300484¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z}trace_0z~trace_1

trace_0
trace_12Þ
L__inference_time_distributed_layer_call_and_return_conditional_losses_300508
L__inference_time_distributed_layer_call_and_return_conditional_losses_300532¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

6kernel
7bias
!_jit_compiled_convolution_op"
_tf_keras_layer
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
ç
trace_0
trace_12¬
3__inference_time_distributed_1_layer_call_fn_300541
3__inference_time_distributed_1_layer_call_fn_300550¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12â
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_300574
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_300598¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

8kernel
9bias
!_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
ç
trace_0
trace_12¬
3__inference_time_distributed_2_layer_call_fn_300603
3__inference_time_distributed_2_layer_call_fn_300608¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
 trace_12â
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_300625
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_300642¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0z trace_1
«
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
ç
¬trace_0
­trace_12¬
3__inference_time_distributed_3_layer_call_fn_300651
3__inference_time_distributed_3_layer_call_fn_300660¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¬trace_0z­trace_1

®trace_0
¯trace_12â
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_300681
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_300702¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0z¯trace_1
Á
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses

:kernel
;bias"
_tf_keras_layer
1:/2time_distributed/kernel
#:!2time_distributed/bias
3:1 2time_distributed_1/kernel
%:# 2time_distributed_1/bias
,:*	@@2time_distributed_3/kernel
%:#@2time_distributed_3/bias
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
+__inference_sequential_layer_call_fn_299437input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
+__inference_sequential_layer_call_fn_300147inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
+__inference_sequential_layer_call_fn_300164inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
+__inference_sequential_layer_call_fn_299516input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_300214inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_300264inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_299544input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_299572input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
ç
»trace_0
¼trace_12¬
3__inference_time_distributed_4_layer_call_fn_300711
3__inference_time_distributed_4_layer_call_fn_300720¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z»trace_0z¼trace_1

½trace_0
¾trace_12â
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_300742
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_300764¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z½trace_0z¾trace_1
Á
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses

ekernel
fbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ç
Êtrace_0
Ëtrace_12¬
3__inference_time_distributed_5_layer_call_fn_300769
3__inference_time_distributed_5_layer_call_fn_300774¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÊtrace_0zËtrace_1

Ìtrace_0
Ítrace_12â
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_300801
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_300828¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÌtrace_0zÍtrace_1
«
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"
_tf_keras_layer
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
ç
Ùtrace_0
Útrace_12¬
3__inference_time_distributed_6_layer_call_fn_300837
3__inference_time_distributed_6_layer_call_fn_300846¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÙtrace_0zÚtrace_1

Ûtrace_0
Ütrace_12â
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_300883
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_300920¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÛtrace_0zÜtrace_1
ä
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses

gkernel
hbias
!ã_jit_compiled_convolution_op"
_tf_keras_layer
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ç
étrace_0
êtrace_12¬
3__inference_time_distributed_7_layer_call_fn_300929
3__inference_time_distributed_7_layer_call_fn_300938¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zétrace_0zêtrace_1

ëtrace_0
ìtrace_12â
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_300974
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_301010¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zëtrace_0zìtrace_1
ä
í	variables
îtrainable_variables
ïregularization_losses
ð	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses

ikernel
jbias
!ó_jit_compiled_convolution_op"
_tf_keras_layer
,:*	 @2time_distributed_4/kernel
&:$@2time_distributed_4/bias
3:1 2time_distributed_6/kernel
%:#2time_distributed_6/bias
3:12time_distributed_7/kernel
%:#2time_distributed_7/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÿBü
-__inference_sequential_1_layer_call_fn_299995input_2"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_sequential_1_layer_call_fn_300281inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_sequential_1_layer_call_fn_300298inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
-__inference_sequential_1_layer_call_fn_300074input_2"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_1_layer_call_and_return_conditional_losses_300382inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_1_layer_call_and_return_conditional_losses_300466inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_1_layer_call_and_return_conditional_losses_300102input_2"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_1_layer_call_and_return_conditional_losses_300130input_2"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bÿ
1__inference_time_distributed_layer_call_fn_300475inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
1__inference_time_distributed_layer_call_fn_300484inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
L__inference_time_distributed_layer_call_and_return_conditional_losses_300508inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
L__inference_time_distributed_layer_call_and_return_conditional_losses_300532inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
ùtrace_02Î
'__inference_conv2d_layer_call_fn_301019¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zùtrace_0

útrace_02é
B__inference_conv2d_layer_call_and_return_conditional_losses_301030¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zútrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_time_distributed_1_layer_call_fn_300541inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
3__inference_time_distributed_1_layer_call_fn_300550inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_300574inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_300598inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
trace_02Ð
)__inference_conv2d_1_layer_call_fn_301039¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ë
D__inference_conv2d_1_layer_call_and_return_conditional_losses_301050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_time_distributed_2_layer_call_fn_300603inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
3__inference_time_distributed_2_layer_call_fn_300608inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_300625inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_300642inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_flatten_layer_call_fn_301055¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_flatten_layer_call_and_return_conditional_losses_301061¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_time_distributed_3_layer_call_fn_300651inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
3__inference_time_distributed_3_layer_call_fn_300660inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_300681inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_300702inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
ì
trace_02Í
&__inference_dense_layer_call_fn_301070¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02è
A__inference_dense_layer_call_and_return_conditional_losses_301080¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
'
O0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_time_distributed_4_layer_call_fn_300711inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
3__inference_time_distributed_4_layer_call_fn_300720inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_300742inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_300764inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_dense_1_layer_call_fn_301089¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_dense_1_layer_call_and_return_conditional_losses_301100¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
'
V0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_time_distributed_5_layer_call_fn_300769inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
3__inference_time_distributed_5_layer_call_fn_300774inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_300801inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_300828inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_reshape_layer_call_fn_301105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_reshape_layer_call_and_return_conditional_losses_301119¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
'
]0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_time_distributed_6_layer_call_fn_300837inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
3__inference_time_distributed_6_layer_call_fn_300846inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_300883inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_300920inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
÷
£trace_02Ø
1__inference_conv2d_transpose_layer_call_fn_301128¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z£trace_0

¤trace_02ó
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_301166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¤trace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
'
d0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_time_distributed_7_layer_call_fn_300929inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
3__inference_time_distributed_7_layer_call_fn_300938inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_300974inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_301010inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
í	variables
îtrainable_variables
ïregularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
ù
ªtrace_02Ú
3__inference_conv2d_transpose_1_layer_call_fn_301175¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zªtrace_0

«trace_02õ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_301212¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z«trace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_conv2d_layer_call_fn_301019inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_conv2d_layer_call_and_return_conditional_losses_301030inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_conv2d_1_layer_call_fn_301039inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_301050inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_flatten_layer_call_fn_301055inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_flatten_layer_call_and_return_conditional_losses_301061inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
&__inference_dense_layer_call_fn_301070inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õBò
A__inference_dense_layer_call_and_return_conditional_losses_301080inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_1_layer_call_fn_301089inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_dense_1_layer_call_and_return_conditional_losses_301100inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_reshape_layer_call_fn_301105inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_reshape_layer_call_and_return_conditional_losses_301119inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
åBâ
1__inference_conv2d_transpose_layer_call_fn_301128inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bý
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_301166inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
çBä
3__inference_conv2d_transpose_1_layer_call_fn_301175inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_301212inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ´
D__inference_conv2d_1_layer_call_and_return_conditional_losses_301050l897¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv2d_1_layer_call_fn_301039_897¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ ²
B__inference_conv2d_layer_call_and_return_conditional_losses_301030l677¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
'__inference_conv2d_layer_call_fn_301019_677¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿã
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_301212ijI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 »
3__inference_conv2d_transpose_1_layer_call_fn_301175ijI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_301166ghI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
1__inference_conv2d_transpose_layer_call_fn_301128ghI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_1_layer_call_and_return_conditional_losses_301100]ef/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ@
 |
(__inference_dense_1_layer_call_fn_301089Pef/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ@¢
A__inference_dense_layer_call_and_return_conditional_losses_301080]:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 z
&__inference_dense_layer_call_fn_301070P:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¨
C__inference_flatten_layer_call_and_return_conditional_losses_301061a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_flatten_layer_call_fn_301055T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ@¨
C__inference_reshape_layer_call_and_return_conditional_losses_301119a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
(__inference_reshape_layer_call_fn_301105T0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ Å
H__inference_sequential_1_layer_call_and_return_conditional_losses_300102yefghij<¢9
2¢/
%"
input_2ÿÿÿÿÿÿÿÿÿ 
p 

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ
 Å
H__inference_sequential_1_layer_call_and_return_conditional_losses_300130yefghij<¢9
2¢/
%"
input_2ÿÿÿÿÿÿÿÿÿ 
p

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_sequential_1_layer_call_and_return_conditional_losses_300382xefghij;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_sequential_1_layer_call_and_return_conditional_losses_300466xefghij;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "1¢.
'$
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_1_layer_call_fn_299995lefghij<¢9
2¢/
%"
input_2ÿÿÿÿÿÿÿÿÿ 
p 

 
ª "$!ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_300074lefghij<¢9
2¢/
%"
input_2ÿÿÿÿÿÿÿÿÿ 
p

 
ª "$!ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_300281kefghij;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "$!ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_300298kefghij;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "$!ÿÿÿÿÿÿÿÿÿÃ
F__inference_sequential_layer_call_and_return_conditional_losses_299544y6789:;D¢A
:¢7
-*
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 Ã
F__inference_sequential_layer_call_and_return_conditional_losses_299572y6789:;D¢A
:¢7
-*
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 Â
F__inference_sequential_layer_call_and_return_conditional_losses_300214x6789:;C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 Â
F__inference_sequential_layer_call_and_return_conditional_losses_300264x6789:;C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_sequential_layer_call_fn_299437l6789:;D¢A
:¢7
-*
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@
+__inference_sequential_layer_call_fn_299516l6789:;D¢A
:¢7
-*
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ@
+__inference_sequential_layer_call_fn_300147k6789:;C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@
+__inference_sequential_layer_call_fn_300164k6789:;C¢@
9¢6
,)
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ@á
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_30057489L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 á
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_30059889L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¹
3__inference_time_distributed_1_layer_call_fn_30054189L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¹
3__inference_time_distributed_1_layer_call_fn_30055089L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ö
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_300625L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ö
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_300642L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ­
3__inference_time_distributed_2_layer_call_fn_300603vL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@­
3__inference_time_distributed_2_layer_call_fn_300608vL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ñ
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_300681:;E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ñ
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_300702:;E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ©
3__inference_time_distributed_3_layer_call_fn_300651r:;E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@©
3__inference_time_distributed_3_layer_call_fn_300660r:;E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ñ
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_300742efD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ñ
N__inference_time_distributed_4_layer_call_and_return_conditional_losses_300764efD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ©
3__inference_time_distributed_4_layer_call_fn_300711refD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@©
3__inference_time_distributed_4_layer_call_fn_300720refD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ö
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_300801E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ö
N__inference_time_distributed_5_layer_call_and_return_conditional_losses_300828E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ­
3__inference_time_distributed_5_layer_call_fn_300769vE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ­
3__inference_time_distributed_5_layer_call_fn_300774vE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ á
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_300883ghL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 á
N__inference_time_distributed_6_layer_call_and_return_conditional_losses_300920ghL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
3__inference_time_distributed_6_layer_call_fn_300837ghL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
3__inference_time_distributed_6_layer_call_fn_300846ghL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_300974ijL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 á
N__inference_time_distributed_7_layer_call_and_return_conditional_losses_301010ijL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
3__inference_time_distributed_7_layer_call_fn_300929ijL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
3__inference_time_distributed_7_layer_call_fn_300938ijL¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿß
L__inference_time_distributed_layer_call_and_return_conditional_losses_30050867L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ß
L__inference_time_distributed_layer_call_and_return_conditional_losses_30053267L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª ":¢7
0-
0&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ·
1__inference_time_distributed_layer_call_fn_30047567L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
1__inference_time_distributed_layer_call_fn_30048467L¢I
B¢?
52
inputs&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "-*&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ