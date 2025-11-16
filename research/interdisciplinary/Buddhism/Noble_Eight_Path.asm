section .data
    Step1 db "Step 1: Right View", 0
    Step1Description db "Develop a correct understanding of the Four Noble Truths and the nature of reality.", 0
    Step2 db "Step 2: Right Intention", 0
    Step2Description db "Cultivate intentions of renunciation, goodwill, and harmlessness.", 0
    Step3 db "Step 3: Right Speech", 0
    Step3Description db "Refrain from lying, divisive speech, harsh language, and idle talk.", 0
    Step4 db "Step 4: Right Action", 0
    Step4Description db "Abstain from killing, stealing, and engaging in sexual misconduct.", 0
    Step5 db "Step 5: Right Livelihood", 0
    Step5Description db "Choose a livelihood that is ethical and supports the principles of the path.", 0
    Step6 db "Step 6: Right Effort", 0
    Step6Description db "Cultivate positive mental states, prevent negative ones, and maintain a balanced effort.", 0
    Step7 db "Step 7: Right Mindfulness", 0
    Step7Description db "Develop awareness and attentiveness to body, feelings, mind, and mental phenomena.", 0
    Step8 db "Step 8: Right Concentration", 0
    Step8Description db "Cultivate the practice of meditation to attain deep states of concentration and insight.", 0

section .text
    global _start

_start:
    ; Display Step 1
    mov eax, 4          ; Syscall for sys_write
    mov ebx, 1          ; File descriptor (stdout)
    mov edx, Step1      ; Address of Step 1
    call print_string

    ; Display Step 1 Description
    mov edx, Step1Description   ; Address of Step 1 Description
    call print_string

    ; Display Step 2
    mov edx, Step2      ; Address of Step 2
    call print_string

    ; Display Step 2 Description
    mov edx, Step2Description   ; Address of Step 2 Description
    call print_string

    ; Display Step 3
    mov edx, Step3      ; Address of Step 3
    call print_string

    ; Display Step 3 Description
    mov edx, Step3Description   ; Address of Step 3 Description
    call print_string

    ; Display Step 4
    mov edx, Step4      ; Address of Step 4
    call print_string

    ; Display Step 4 Description
    mov edx, Step4Description   ; Address of Step 4 Description
    call print_string

    ; Display Step 5
    mov edx, Step5      ; Address of Step 5
    call print_string

    ; Display Step 5 Description
    mov edx, Step5Description   ; Address of Step 5 Description
    call print_string

    ; Display Step 6
    mov edx, Step6      ; Address of Step 6
    call print_string

    ; Display Step 6 Description
    mov edx, Step6Description   ; Address of Step 6 Description
    call print_string

    ; Display Step 7
    mov edx, Step7      ; Address of Step 7
    call print_string

    ; Display Step 7 Description
    mov edx, Step7Description   ; Address of Step 7 Description
    call print_string

    ; Display Step 8
    mov edx, Step8      ; Address of Step 8
    call print_string

    ; Display Step 8 Description
    mov edx, Step8Description   ; Address of Step 8 Description
    call print_string

    ; Exit the program
    mov eax, 1          ; Syscall for sys_exit
    xor ebx, ebx        ; Exit code 0
    int 0x80

print_string:
    ; Function to display a null-terminated string
    ; Input: edx - Address of the string
    ; Output: None
    mov eax, 4          ; Syscall for sys_write
    mov ecx, edx        ; Load the address of the string to ecx
    xor ebx, ebx        ; File descriptor (stdout)
    xor edx, edx        ; String length (0, as it is null-terminated)
    mov dl, 100         ; Set maximum string length to 100
    int 0x80
    ret
