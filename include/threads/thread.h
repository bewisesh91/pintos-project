#ifndef THREADS_THREAD_H
#define THREADS_THREAD_H

#include <debug.h>
#include <list.h>
#include <stdint.h>
#include "threads/interrupt.h"
#include "threads/synch.h"
#ifdef VM
#include "vm/vm.h"
#endif


/* States in a thread's life cycle. */
enum thread_status {
    THREAD_RUNNING,     /* Running thread. */
    THREAD_READY,       /* Not running but ready to run. */
    THREAD_BLOCKED,     /* Waiting for an event to trigger. */
    THREAD_DYING        /* About to be destroyed. */
};

/* Thread identifier type.
   You can redefine this to whatever type you like. */
typedef int tid_t;
#define TID_ERROR ((tid_t) -1)          /* Error value for tid_t. */

/* Thread priorities. */
#define PRI_MIN 0                       /* Lowest priority. */
#define PRI_DEFAULT 31                  /* Default priority. */
#define PRI_MAX 63                      /* Highest priority. */
#define NICE_DEFAULT 0
#define RECENT_CPU_DEFAULT 0
#define LOAD_AVG_DEFAULT 0

/* A kernel thread or user process.
 *
 * Each thread structure is stored in its own 4 kB page.  The
 * thread structure itself sits at the very bottom of the page
 * (at offset 0).  The rest of the page is reserved for the
 * thread's kernel stack, which grows downward from the top of
 * the page (at offset 4 kB).  Here's an illustration:
 *
 *      4 kB +---------------------------------+
 *           |          kernel stack           |
 *           |                |                |
 *           |                |                |
 *           |                V                |
 *           |         grows downward          |
 *           |                                 |
 *           |                                 |
 *           |                                 |
 *           |                                 |
 *           |                                 |
 *           |                                 |
 *           |                                 |
 *           |                                 |
 *           +---------------------------------+
 *           |              magic              |
 *           |            intr_frame           |
 *           |                :                |
 *           |                :                |
 *           |               name              |
 *           |              status             |
 *      0 kB +---------------------------------+
 *
 * The upshot of this is twofold:
 *
 *    1. First, `struct thread' must not be allowed to grow too
 *       big.  If it does, then there will not be enough room for
 *       the kernel stack.  Our base `struct thread' is only a
 *       few bytes in size.  It probably should stay well under 1
 *       kB.
 *
 *    2. Second, kernel stacks must not be allowed to grow too
 *       large.  If a stack overflows, it will corrupt the thread
 *       state.  Thus, kernel functions should not allocate large
 *       structures or arrays as non-static local variables.  Use
 *       dynamic allocation with malloc() or palloc_get_page()
 *       instead.
 *
 * The first symptom of either of these problems will probably be
 * an assertion failure in thread_current(), which checks that
 * the `magic' member of the running thread's `struct thread' is
 * set to THREAD_MAGIC.  Stack overflow will normally change this
 * value, triggering the assertion. */
/* The `elem' member has a dual purpose.  It can be an element in
 * the run queue (thread.c), or it can be an element in a
 * semaphore wait list (synch.c).  It can be used these two ways
 * only because they are mutually exclusive: only a thread in the
 * ready state is on the run queue, whereas only a thread in the
 * blocked state is on a semaphore wait list. */
struct thread {
    /* Owned by thread.c. */
    tid_t tid;                          /* Thread identifier. */
    enum thread_status status;          /* Thread state. */
    char name[16];                      /* Name (for debugging purposes). */
    int priority;                       /* Priority. */

    /* Shared between thread.c and synch.c. */
    struct list_elem elem;              /* List element. */
    struct list_elem allelem;           /* advanced scheduling */
    int64_t wakeup_tick;                /* 깨어날 시간 저장 */

    /* variable for donation*/
    int init_priority; //스레드가 priority를 양도받았다가 다시 반납할 때 원래의 priority를 복원할 수 있도록 고유의 priority 값을 저장
    struct lock *wait_on_lock; // 스레드가 현재 얻기 위해 기다리고 있는 lock으로 스레드는 이 lock이 release되기를 기다린다.
    struct list donations; // 자신에게 priority를 나누어준 스레드들의 리스트
    struct list_elem donation_elem; // list donations을 관리하기 위한 element
    
    /* advanced */
    int nice;
    int recent_cpu;
#ifdef USERPROG
    /* Owned by userprog/process.c. */
    uint64_t *pml4;                     /* Page map level 4 */
#endif
#ifdef VM
    /* Table for whole virtual memory owned by thread. */
    struct supplemental_page_table spt;
#endif

    /* Owned by thread.c. */
    struct intr_frame tf;               /* Information for switching */
    unsigned magic;                     /* Detects stack overflow. */

    /* 자식 프로세스 순회용 리스트 */
    struct list child_list;
    struct list_elem child_elem; 

    /* wait_sema 를 이용하여 자식 프로세스가 종료할때까지 대기함. 종료 상태를 저장 */
    struct semaphore wait_sema;
    int exit_status;

    /* 자식에게 넘겨줄 intr_frame
    fork가 완료될때 까지 부모가 기다리게 하는 forksema
    자식 프로세스 종료상태를 부모가 받을때까지 종료를 대기하게 하는 free_sema */
    struct intr_frame parent_if;
    struct semaphore fork_sema;
    struct semaphore free_sema;

    /* fd table 파일 구조체와 fd index */
    struct file **fdTable;
    int fdIdx;

    int stdin_count;
    int stdout_count;

    /* 현재 실행 중인 파일 */
    struct file *running;


};

/* If false (default), use round-robin scheduler.
   If true, use multi-level feedback queue scheduler.
   Controlled by kernel command-line option "-o mlfqs". */
extern bool thread_mlfqs;



void thread_init (void);
void thread_start (void);

void thread_tick (void);
void thread_print_stats (void);



typedef void thread_func (void *aux);
tid_t thread_create (const char *name, int priority, thread_func *, void *);

void thread_block (void);
void thread_unblock (struct thread *);

struct thread *thread_current (void);
tid_t thread_tid (void);
const char *thread_name (void);

void thread_exit (void) NO_RETURN;
void thread_yield (void);

int thread_get_priority (void);
void thread_set_priority (int);

int thread_get_nice (void);
void thread_set_nice (int);
int thread_get_recent_cpu (void);
int thread_get_load_avg (void);

/* function for alarm*/
void update_next_tick_to_awake(int64_t ticks);
int64_t get_next_tick_to_awake(void);
void thread_sleep(int64_t ticks);
void thread_awake(int64_t ticks);

/* function for priority */

bool thread_compare_priority(const struct list_elem *a,
                              const struct list_elem *b,
                              void *aux UNUSED); // 스레드의 우선순위 비교
void test_max_priority(void); // 첫번째 스레드가 cpu 점유 중인 스레드 보다 우선순위가 높으면 cpu 점유를 양보하는 함수
bool check_preemption();
void do_iret (struct intr_frame *tf);

/* priority donation */
void donate_priority();
void remove_with_lock(struct lock *lock);
void refresh_priority(void);
bool thread_compare_donate_priority(const struct list_elem *l, const struct list_elem *s, void *aux UNUSED);

/* advance */
void mlfqs_calculate_priority(struct thread *t);
void mlfqs_calculate_recent_cpu(struct thread *t);
void mlfqs_calculate_load_avg(void);
void mlfqs_increments_recent_cpu(void);
void mlfqs_recalculate_recent_cpu(void);
void mlfqs_recalculate_recent_cpu(void);

/* fixed_point */
int int_to_fp (int n);
int fp_to_int (int x);
int fp_to_int_round (int x);
int add_fp (int x, int y);
int sub_fp (int x, int y);
int add_mixed (int x, int n);
int sub_mixed (int x, int n);
int mult_fp (int x, int y);
int mult_mixed (int x, int n);
int div_fp (int x, int y);
int div_mixed (int x, int n);

/* 파일 디스크립터 상수 */
#define FDT_PAGES 3
#define FDCOUNT_LIMIT FDT_PAGES * (1<<9)

#endif /* threads/thread.h */

