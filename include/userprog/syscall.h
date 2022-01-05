#ifndef USERPROG_SYSCALL_H
#define USERPROG_SYSCALL_H

void syscall_init (void);

/* 파일 사용시 lock하여 상호배제 구현 */
struct lock file_rw_lock;

#endif /* userprog/syscall.h */
