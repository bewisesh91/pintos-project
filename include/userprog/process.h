#ifndef USERPROG_PROCESS_H
#define USERPROG_PROCESS_H

#include "threads/thread.h"

tid_t process_create_initd (const char *file_name);
tid_t process_fork (const char *name, struct intr_frame *if_);
int process_exec (void *f_name);
int process_wait (tid_t);
void process_exit (void);
void process_activate (struct thread *next);

/* pid를 입력하여 자식프로세스인지 확인하여 맞다면 thread 구조체 반환 */
struct thread *get_child_with_pid(int pid);

#endif /* userprog/process.h */
