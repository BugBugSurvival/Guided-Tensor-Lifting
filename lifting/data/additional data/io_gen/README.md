# Input-Output generation

Here we give instructions on how to automatically generate an IO set for a user program.

## Value profile
If you want to automatically generate IO examples to a C function, you first need to specify a value profile file. A value profile is a JSON file containing specification of array dimensions and values of integer variables. The value profile must contain an entry for each argument of the C function. For example, consider the program below:

```c
void scale_array(int *a, int s, int n){
  int i;
  for(i = 0; i < n; ++i){
    a[i] *= s;
  }
}
```

a possible value profile is shown below:

```yaml
{
  "a": 20,
  "s": 3,
  "n": 20
}
```
In the profile above, array `a` is specified as having 20 elements. Variable `n` has the value 20 since is used as control variable for the loop that traverses `a`. Variable `s` will have the value 3 in the generated IO examples.



### additional notes
Create a value profile file. The file should have an entry mapping each argument in the function to a number

 - If the variable is a pointer, the number is its size
 - If the variable is an integer, the number is its value
 
 
To facilitate distribution, tensors should have 4096 elements. Therefore, if the pointer is a 1D array, it represents a 1x4096 vector. If it is a matrix, 64x64 and so on. However, not all pointers should be assigned 4096 depending on the computation. The dimensions should match. For example, a(i) = b(i,j) * c(j), we should assign a -> 64, b -> 4096, c-> 64

When creating the value profile, always assign 4096 for pointers and you have to be careful about the values for integers. 

 - If the integer represents the size of an input (for example, if it is used in a loop), assign it according to the order of the pointers accessed in the loop. 
   - If the pointer is 1d (p[]), assign 2096
   - if the pointer is 2d(p[][]), assign 64
   - if the pointer is 3d (p[][][]), assign 16, and so on.
   
 - If the integer does not represent the size of an input, you can assign any value to it
 
Example of value profile file for `atax.c`

```
void atax(int _PB_NX, int _PB_NY,
                 int** A, int *x, int* y, int* tmp
		 )
```

```		 
{
  "A" : 4096,
  "x" : 64,
  "y" : 64,
  "tmp: 64,
  "_PB_NX": 64,
  "_PB_NY": 64
}
```

Always assign 4096 to the largest tensor and adjust the others. In this example, A is a matrix while x, y, and tmp are 1d arrays. So you should assign 4096 to A and 64 to the rest. Why 64? Because A is a 64x64 matrix and the dimensions should match. You have to analyze case by case to check if the dimensions match. _PB_NX and _PB_NY represent sizes, so assign 64 as well. 

Also, (unless the function is returning a non-zero value) you should tell the io generator which variable is the output. In this case the output is "y". Identifyhing the output variable is a manual process and needs to be done for each benchmark individually.




## Usage
To generate an IO file, simply run the `io_gen.py` file as shown below.

```
$ python3 io_gen.py -b <path-to-c-program> -vp <path-to-value-profile-file> [-ov output_variable_name] -n numer_of_inputs
```

The script will first parse the input program to get information about its signature. Given said information and the value profile data, a driver will be created. Such driver contains a `main` routine that initialize the required variables, call the original program and prints input and output values. The driver is compiled together with the original program plus the `io_gen.c` file, which contains auxiliary methods. The IO generator scripts then executes the generated binary and save its output to a temporary file which is parsed and a final JSON is produced contaning the IO set.

The script takes 4 arguments, of which 3 are mandatory:

* `-b, --benchmark`: benchmark to generate IO for
* `-vp, --valueprofile`: value profile file
* `-n, --ninputs`: number of IO samples to be generated
* `-ov, --outvar`: variable that corresponds to the output

The last argument, `-ov` is only mandatory if the program does not explicitly returns the output, e.g., its output is done by changing the values in the region pointed by a pointer that is a parameter to the program.

Using the program and value profile file above, we can generate an IO set with 2 samples by executing the command:

```
$ python3 io_gen.py -b <path-to>/scale_array.c -vp <path-to>/scale_array_value_profile.json -n 2 -ov a
```

Notice that we need to pass the argument `-ov a` as `scale_array` does not explicitly returns its output: it changes the values pointed by pointer `a`. The command below produces the file `scale_array_io.json` with 2 different IO samples, as shown below

```yaml
[
  {
    "input": {
      "a": [
        " 20",
        " 98 49 66 43 58 87 91 80 22 23 9 8 11 2 32 53 9 35 14 42"
      ],
      "n": [
        " 1",
        " 20"
      ],
      "s": [
        " 1",
        " 3"
      ]
    },
    "output": {
      "a": [
        " 20",
        " 294 147 198 129 174 261 273 240 66 69 27 24 33 6 96 159 27 105 42 126"
      ]
    }
  },
  {
    "input": {
      "a": [
        " 20",
        " 72 31 47 14 71 74 45 60 72 72 28 21 20 45 63 30 83 5 61 4"
      ],
      "n": [
        " 1",
        " 20"
      ],
      "s": [
        " 1",
        " 3"
      ]
    },
    "output": {
      "a": [
        " 20",
        " 216 93 141 42 213 222 135 180 216 216 84 63 60 135 189 90 249 15 183 12"
      ]
    }
  }
]
```

On each sample, input and output values are separated. Each variable entry contains two fields. The first one is the length of the variable, always 1 for scalars, and the second holds the actual values.


## Restrictions

* the input programs must have a single function inside
* only integer programs are supported (including pointers and arrays)
* for tensors with order bigger than 2, the specified values must for square tensors. For example, matrices must be `n x n`, 3D matrices must be `n x n x n`
* it is important to define consistent values in the value profile file. For example, if the input program is computing a matrix-vector product, the dimensions multiplied must be compatible.


