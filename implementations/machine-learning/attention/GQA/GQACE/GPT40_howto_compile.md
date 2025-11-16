To compile the five C programs on an Ubuntu 22.04 server, you can follow these steps. I'll assume you have a text editor and GCC (GNU Compiler Collection) installed on your target server.

### **Step 1: Ensure GCC is Installed**
Ubuntu typically comes with GCC pre-installed. If not, you can install it:
```bash
sudo apt update
sudo apt install build-essential
```

This will install GCC and other essential tools for building programs.

---

### **Step 2: Create the C Source Files**
Save each example as a separate `.c` file. For example:
- Example 1: `example1_gqa.c`
- Example 2: `example2_dynamic_key_distribution.c`
- Example 3: `example3_perturbed_gqa.c`
- Example 4: `example4_dynamic_ema.c`
- Example 5: `example5_varying_heads.c`

You can use a text editor like `nano` or `vim` to create and edit these files:
```bash
nano example1_gqa.c
```
Paste the corresponding C code into the file, then save it.

---

### **Step 3: Compile the Programs**
To compile each program, use the GCC command:
```bash
gcc -o example1_gqa example1_gqa.c
gcc -o example2_dynamic_key_distribution example2_dynamic_key_distribution.c
gcc -o example3_perturbed_gqa example3_perturbed_gqa.c
gcc -o example4_dynamic_ema example4_dynamic_ema.c
gcc -o example5_varying_heads example5_varying_heads.c
```

This command compiles each C file into an executable binary with the same name as the output file specified after `-o`.

---

### **Step 4: Run the Compiled Programs**
Once compiled, you can execute each program like this:
```bash
./example1_gqa
./example2_dynamic_key_distribution
./example3_perturbed_gqa
./example4_dynamic_ema
./example5_varying_heads
```

---

### **Step 5: Debugging Compilation Errors**
If any of the programs fail to compile, GCC will output error messages. Some common fixes:
- Ensure all `#include` directives are present.
- Check for typos in the code.
- Ensure that the file was saved correctly and doesn’t contain unsupported characters.

---

### **Optional: Automate the Process with a Makefile**
You can automate the compilation process using a Makefile:
1. Create a file named `Makefile`:
```makefile
all: example1_gqa example2_dynamic_key_distribution example3_perturbed_gqa example4_dynamic_ema example5_varying_heads

example1_gqa: example1_gqa.c
	gcc -o example1_gqa example1_gqa.c

example2_dynamic_key_distribution: example2_dynamic_key_distribution.c
	gcc -o example2_dynamic_key_distribution example2_dynamic_key_distribution.c

example3_perturbed_gqa: example3_perturbed_gqa.c
	gcc -o example3_perturbed_gqa example3_perturbed_gqa.c

example4_dynamic_ema: example4_dynamic_ema.c
	gcc -o example4_dynamic_ema example4_dynamic_ema.c

example5_varying_heads: example5_varying_heads.c
	gcc -o example5_varying_heads example5_varying_heads.c

clean:
	rm -f example1_gqa example2_dynamic_key_distribution example3_perturbed_gqa example4_dynamic_ema example5_varying_heads
```

2. Run `make` to compile all programs:
```bash
make
```

3. Use `make clean` to clean up the compiled binaries:
```bash
make clean
```

---

### **Summary**
- Install GCC if necessary.
- Save the C programs in individual files.
- Compile them with GCC.
- Run the executables directly.
- Optionally, use a Makefile for automation.

This process ensures you can compile and test all five programs efficiently on your Ubuntu 22.04 server.


Here's a bash script that automates the creation of the C source files for all five examples. Save this script as `generate_c_sources.sh` and run it on your Ubuntu 22.04 server.

```bash
#!/bin/bash

# Define source code for each example
EXAMPLE_1='
#include <stdio.h>

#define NUM_QUERIES 6
#define NUM_GROUPS 2
#define DIM 3

void mean_pooling(float keys[NUM_QUERIES][DIM], float pooled_keys[NUM_GROUPS][DIM]) {
    for (int g = 0; g < NUM_GROUPS; g++) {
        for (int d = 0; d < DIM; d++) {
            pooled_keys[g][d] = 0;
            for (int q = g * (NUM_QUERIES / NUM_GROUPS); q < (g + 1) * (NUM_QUERIES / NUM_GROUPS); q++) {
                pooled_keys[g][d] += keys[q][d];
            }
            pooled_keys[g][d] /= (NUM_QUERIES / NUM_GROUPS);
        }
    }
}

int main() {
    float queries[NUM_QUERIES][DIM] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {2, 4, 6}, {8, 10, 12}, {3, 6, 9}};
    float keys[NUM_QUERIES][DIM] = {{1, 0, 1}, {0, 1, 0}, {1, 1, 1}, {0, 0, 1}, {1, 0, 0}, {0, 1, 1}};
    float pooled_keys[NUM_GROUPS][DIM];

    mean_pooling(keys, pooled_keys);

    printf("Pooled Keys:\\n");
    for (int g = 0; g < NUM_GROUPS; g++) {
        for (int d = 0; d < DIM; d++) {
            printf("%.2f ", pooled_keys[g][d]);
        }
        printf("\\n");
    }
    return 0;
}'

EXAMPLE_2='
#include <stdio.h>
#include <math.h>

#define NUM_KEYS 4
#define DIM 3

float calculate_norm(float key[DIM]) {
    float norm = 0;
    for (int i = 0; i < DIM; i++) {
        norm += key[i] * key[i];
    }
    return sqrt(norm);
}

void distribute_queries(float keys[NUM_KEYS][DIM], int query_distribution[NUM_KEYS], int total_queries) {
    float norms[NUM_KEYS], norm_sum = 0;

    for (int i = 0; i < NUM_KEYS; i++) {
        norms[i] = calculate_norm(keys[i]);
        norm_sum += norms[i];
    }

    for (int i = 0; i < NUM_KEYS; i++) {
        query_distribution[i] = (int)(total_queries * (norms[i] / norm_sum));
    }
}

int main() {
    float keys[NUM_KEYS][DIM] = {{1, 2, 3}, {4, 5, 6}, {1, 1, 1}, {3, 3, 3}};
    int query_distribution[NUM_KEYS];
    int total_queries = 10;

    distribute_queries(keys, query_distribution, total_queries);

    printf("Query Distribution:\\n");
    for (int i = 0; i < NUM_KEYS; i++) {
        printf("Key %d: %d queries\\n", i, query_distribution[i]);
    }

    return 0;
}'

EXAMPLE_3='
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_GROUPS 2
#define DIM 3

void add_noise(float attention_map[NUM_GROUPS][DIM], float noise_level) {
    srand(time(0));
    for (int g = 0; g < NUM_GROUPS; g++) {
        for (int d = 0; d < DIM; d++) {
            float noise = ((float)rand() / RAND_MAX) * noise_level - (noise_level / 2);
            attention_map[g][d] += noise;
        }
    }
}

int main() {
    float attention_map[NUM_GROUPS][DIM] = {{0.8, 0.1, 0.1}, {0.4, 0.3, 0.3}};
    float noise_level = 0.05;

    printf("Original Attention Map:\\n");
    for (int g = 0; g < NUM_GROUPS; g++) {
        for (int d = 0; d < DIM; d++) {
            printf("%.2f ", attention_map[g][d]);
        }
        printf("\\n");
    }

    add_noise(attention_map, noise_level);

    printf("\\nAttention Map with Noise:\\n");
    for (int g = 0; g < NUM_GROUPS; g++) {
        for (int d = 0; d < DIM; d++) {
            printf("%.2f ", attention_map[g][d]);
        }
        printf("\\n");
    }

    return 0;
}'

EXAMPLE_4='
#include <stdio.h>

#define NUM_KEYS 4
#define ALPHA 0.8

void update_ema(float current_norms[NUM_KEYS], float ema[NUM_KEYS]) {
    for (int i = 0; i < NUM_KEYS; i++) {
        ema[i] = ALPHA * current_norms[i] + (1 - ALPHA) * ema[i];
    }
}

int main() {
    float current_norms[NUM_KEYS] = {1.2, 0.8, 1.5, 0.7};
    float ema[NUM_KEYS] = {1.0, 1.0, 1.0, 1.0};

    update_ema(current_norms, ema);

    printf("Updated EMA Norms:\\n");
    for (int i = 0; i < NUM_KEYS; i++) {
        printf("Key %d: %.2f\\n", i, ema[i]);
    }

    return 0;
}'

EXAMPLE_5='
#include <stdio.h>

#define MAX_HEADS 8
#define DIM 3

void scale_attention(float attention[MAX_HEADS][DIM], int num_heads) {
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < DIM; d++) {
            attention[h][d] /= num_heads;
        }
    }
}

int main() {
    float attention[MAX_HEADS][DIM] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}};
    int num_heads = 4;

    printf("Original Attention Map:\\n");
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < DIM; d++) {
            printf("%.2f ", attention[h][d]);
        }
        printf("\\n");
    }

    scale_attention(attention, num_heads);

    printf("\\nScaled Attention Map:\\n");
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < DIM; d++) {
            printf("%.2f ", attention[h][d]);
        }
        printf("\\n");
    }

    return 0;
}'

# Create the C files
echo "$EXAMPLE_1" > example1_gqa.c
echo "$EXAMPLE_2" > example2_dynamic_key_distribution.c
echo "$EXAMPLE_3" > example3_perturbed_gqa.c
echo "$EXAMPLE_4" > example4_dynamic_ema.c
echo "$EXAMPLE_5" > example5_varying_heads.c

# Notify user
echo "C source files have been generated:"
ls -1 example*.c
```

### How to Use the Script
1. Save the script to a file, e.g., `generate_c_sources.sh`.
2. Make the script executable:
   ```bash
   chmod +x generate_c_sources.sh
   ```
3. Run the script:
   ```bash
   ./generate_c_sources.sh
   ```
4. This will generate the C source files (`example1_gqa.c`, etc.) in the current directory.

Now, you can proceed to compile and run these C programs as needed!
