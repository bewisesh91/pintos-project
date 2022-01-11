#include "threads/thread.h"
#include <debug.h>
#include <stddef.h>
#include <random.h>
#include <stdio.h>
#include <string.h>
#include "threads/flags.h"
#include "threads/interrupt.h"
#include "threads/intr-stubs.h"
#include "threads/palloc.h"
#include "threads/synch.h"
#include "threads/vaddr.h"
#include "intrinsic.h"
#ifdef USERPROG
#include "userprog/process.h"
#endif

/* Random value for struct thread's `magic' member.
   Used to detect stack overflow.  See the big comment at the top
   of thread.h for details. */
#define THREAD_MAGIC 0xcd6abf4b

/* Random value for basic thread
   Do not modify this value. */
#define THREAD_BASIC 0xd42df210

#define F (1 << 14)
#define INT_MAX ((1 << 31) - 1)
#define INT_MIN (-(1 << 31))

/* List of processes in THREAD_READY state, that is, processes
   that are ready to run but not actually running. */
static struct list ready_list;
static struct list all_list; // 모든 thread를 넣기위한 list

/* Idle thread. */
static struct thread *idle_thread;

/* Initial thread, the thread running init.c:main(). */
static struct thread *initial_thread;

/* Lock used by allocate_tid(). */
static struct lock tid_lock;

/* Thread destruction requests */
static struct list destruction_req;

/* Statistics. */
static long long idle_ticks;    /* # of timer ticks spent idle. */
static long long kernel_ticks;  /* # of timer ticks in kernel threads. */
static long long user_ticks;    /* # of timer ticks in user programs. */

static struct list sleep_list; /* 대기할 thread를 담아둘 list*/
static int64_t next_tick_to_awake; /* 가장 먼저 일어날 친구가 일어날 시각 변수 */

/* Scheduling. */
#define TIME_SLICE 4            /* # of timer ticks to give each thread. */
static unsigned thread_ticks;   /* # of timer ticks since last yield. */

/* If false (default), use round-robin scheduler.
   If true, use multi-level feedback queue scheduler.
   Controlled by kernel command-line option "-o mlfqs". */
bool thread_mlfqs;

static void kernel_thread (thread_func *, void *aux);

static void idle (void *aux UNUSED);
static struct thread *next_thread_to_run (void);
static void init_thread (struct thread *, const char *name, int priority);
static void do_schedule(int status);
static void schedule (void);
static tid_t allocate_tid (void);



/* Returns true if T appears to point to a valid thread. */
#define is_thread(t) ((t) != NULL && (t)->magic == THREAD_MAGIC)

/* Returns the running thread.
 * Read the CPU's stack pointer `rsp', and then round that
 * down to the start of a page.  Since `struct thread' is
 * always at the beginning of a page and the stack pointer is
 * somewhere in the middle, this locates the curent thread. */
#define running_thread() ((struct thread *) (pg_round_down (rrsp ())))


// Global descriptor table for the thread_start.
// Because the gdt will be setup after the thread_init, we should
// setup temporal gdt first.
static uint64_t gdt[3] = { 0, 0x00af9a000000ffff, 0x00cf92000000ffff };
static int load_avg;

/* Initializes the threading system by transforming the code
   that's currently running into a thread.  This can't work in
   general and it is possible in this case only because loader.S
   was careful to put the bottom of the stack at a page boundary.

   Also initializes the run queue and the tid lock.

   After calling this function, be sure to initialize the page
   allocator before trying to create any threads with
   thread_create().

   It is not safe to call thread_current() until this function
   finishes. */
void
thread_init (void) {
    ASSERT (intr_get_level () == INTR_OFF);

    /* Reload the temporal gdt for the kernel
     * This gdt does not include the user context.
     * The kernel will rebuild the gdt with user context, in gdt_init (). */
    struct desc_ptr gdt_ds = {
        .size = sizeof (gdt) - 1,
        .address = (uint64_t) gdt
    };
    lgdt (&gdt_ds);

    /* Init the global thread context */
    lock_init (&tid_lock);
    list_init (&ready_list);
    list_init (&destruction_req);
    list_init (&sleep_list); // assign1
    list_init (&all_list); // advance

    /* Set up a thread structure for the running thread. */
    initial_thread = running_thread ();
    init_thread (initial_thread, "main", PRI_DEFAULT);
    initial_thread->status = THREAD_RUNNING;
    initial_thread->tid = allocate_tid ();
}

/* Starts preemptive thread scheduling by enabling interrupts.
   Also creates the idle thread. */
void
thread_start (void) {
    /* Create the idle thread. */
    struct semaphore idle_started;
    sema_init (&idle_started, 0);
    thread_create ("idle", PRI_MIN, idle, &idle_started);

    /* Start preemptive thread scheduling. */
    intr_enable ();
    load_avg = LOAD_AVG_DEFAULT;

    /* Wait for the idle thread to initialize idle_thread. */
    sema_down (&idle_started);
}

/* Called by the timer interrupt handler at each timer tick.
   Thus, this function runs in an external interrupt context. */
void
thread_tick (void) {
    struct thread *t = thread_current ();

    /* Update statistics. */
    if (t == idle_thread)
        idle_ticks++;
#ifdef USERPROG
    else if (t->pml4 != NULL)
        user_ticks++;
#endif
    else
        kernel_ticks++;

    /* Enforce preemption. */
    if (++thread_ticks >= TIME_SLICE)
        intr_yield_on_return ();
}

/* Prints thread statistics. */
void
thread_print_stats (void) {
    printf ("Thread: %lld idle ticks, %lld kernel ticks, %lld user ticks\n",
            idle_ticks, kernel_ticks, user_ticks);
}

/* Creates a new kernel thread named NAME with the given initial
   PRIORITY, which executes FUNCTION passing AUX as the argument,
   and adds it to the ready queue.  Returns the thread identifier
   for the new thread, or TID_ERROR if creation fails.

   If thread_start() has been called, then the new thread may be
   scheduled before thread_create() returns.  It could even exit
   before thread_create() returns.  Contrariwise, the original
   thread may run for any amount of time before the new thread is
   scheduled.  Use a semaphore or some other form of
   synchronization if you need to ensure ordering.

   The code provided sets the new thread's `priority' member to
   PRIORITY, but no actual priority scheduling is implemented.
   Priority scheduling is the goal of Problem 1-3. */
tid_t
thread_create (const char *name, int priority,
        thread_func *function, void *aux) {
    struct thread *t;
    tid_t tid;

    ASSERT (function != NULL);

    /* Allocate thread. */
    t = palloc_get_page (PAL_ZERO);
    if (t == NULL)
        return TID_ERROR;

    /* Initialize thread. */
    init_thread (t, name, priority);
    tid = t->tid = allocate_tid ();

    /* 현재 스레드의 자식 리스트에 새로 생성한 스레드 추가 */
    struct thread *curr = thread_current();
    list_push_back(&curr->child_list,&t->child_elem);

    /* 파일 디스크립터 초기화 */
    t->fdTable = palloc_get_multiple(PAL_ZERO,FDT_PAGES);
    if(t->fdTable == NULL)
        return TID_ERROR;
    t->fdIdx = 2;
    t->fdTable[0] = 1;
    t->fdTable[1] = 2;

    t->stdin_count = 1;
    t->stdout_count = 1;

    /* Call the kernel_thread if it scheduled.
     * Note) rdi is 1st argument, and rsi is 2nd argument. */
    t->tf.rip = (uintptr_t) kernel_thread;
    t->tf.R.rdi = (uint64_t) function;
    t->tf.R.rsi = (uint64_t) aux;
    t->tf.ds = SEL_KDSEG;
    t->tf.es = SEL_KDSEG;
    t->tf.ss = SEL_KDSEG;
    t->tf.cs = SEL_KCSEG;
    t->tf.eflags = FLAG_IF;

    /* Add to run queue. */
    thread_unblock (t);

    /* 만약 새로 만든 thread의 우선순위가 현재 실행되고 있는 thread보다 높으면 yields*/
    if(check_preemption()) thread_yield();

    return tid;
}

/* Puts the current thread to sleep.  It will not be scheduled
   again until awoken by thread_unblock().

   This function must be called with interrupts turned off.  It
   is usually a better idea to use one of the synchronization
   primitives in synch.h. */
void
thread_block (void) {
    ASSERT (!intr_context ());
    ASSERT (intr_get_level () == INTR_OFF);
    thread_current ()->status = THREAD_BLOCKED;
    schedule ();
}

/* Transitions a blocked thread T to the ready-to-run state.
   This is an error if T is not blocked.  (Use thread_yield() to
   make the running thread ready.)

   This function does not preempt the running thread.  This can
   be important: if the caller had disabled interrupts itself,
   it may expect that it can atomically unblock a thread and
   update other data. */
void
thread_unblock (struct thread *t) {
    enum intr_level old_level;

    ASSERT (is_thread (t));

    old_level = intr_disable ();
    ASSERT (t->status == THREAD_BLOCKED);
    // list_push_back (&ready_list, &t->elem);
    list_insert_ordered (&ready_list, &t->elem, thread_compare_priority, 0);
    t->status = THREAD_READY;
    intr_set_level (old_level);
}

/* Returns the name of the running thread. */
const char *
thread_name (void) {
    return thread_current ()->name;
}

/* Returns the running thread.
   This is running_thread() plus a couple of sanity checks.
   See the big comment at the top of thread.h for details. */
struct thread *
thread_current (void) {
    struct thread *t = running_thread ();

    /* Make sure T is really a thread.
       If either of these assertions fire, then your thread may
       have overflowed its stack.  Each thread has less than 4 kB
       of stack, so a few big automatic arrays or moderate
       recursion can cause stack overflow. */
    ASSERT (is_thread (t));
    ASSERT (t->status == THREAD_RUNNING);

    return t;
}

/* Returns the running thread's tid. */
tid_t
thread_tid (void) {
    return thread_current ()->tid;
}

/* Deschedules the current thread and destroys it.  Never
   returns to the caller. */
void
thread_exit (void) {
    ASSERT (!intr_context ());

#ifdef USERPROG
    process_exit ();
#endif

    /* Just set our status to dying and schedule another process.
       We will be destroyed during the call to schedule_tail(). */
    intr_disable ();
    list_remove(&thread_current()->allelem); // 전부 사용한 thread는 all_list에서 제거
    do_schedule (THREAD_DYING);
    NOT_REACHED ();
}

/* Yields the CPU.  The current thread is not put to sleep and
   may be scheduled again immediately at the scheduler's whim. */
void
thread_yield (void) {
    struct thread *curr = thread_current ();
    enum intr_level old_level;

    ASSERT (!intr_context ());

    old_level = intr_disable ();
    if (curr != idle_thread)
        // list_push_back (&ready_list, &curr->elem);
        list_insert_ordered (&ready_list, &curr->elem, thread_compare_priority, 0);

    do_schedule (THREAD_READY);
    intr_set_level (old_level);
}

/* Sets the current thread's priority to NEW_PRIORITY. */
void
thread_set_priority (int new_priority) {
    if(thread_mlfqs) return;
    thread_current () -> init_priority = new_priority;
    refresh_priority();
    if(check_preemption()) thread_yield();
}

/* Returns the current thread's priority. */
int
thread_get_priority (void) {
    enum intr_level old_level = intr_disable();

	int ret = thread_current()->priority;
	
    intr_set_level(old_level);
	return ret;
}

/* Sets the current thread's nice value to NICE. */
void
thread_set_nice (int nice) {
    enum intr_level old_level = intr_disable();
    
    thread_current() -> nice = nice;
    mlfqs_calculate_priority(thread_current());

    if(check_preemption()) thread_yield();

    intr_set_level(old_level);
}

/* Returns the current thread's nice value. */
int
thread_get_nice (void) {
    /* TODO: Your implementation goes here */
    enum intr_level old_level = intr_disable();
    int nice = thread_current() -> nice;
    intr_set_level(old_level);
    return nice;
}

/* Returns 100 times the system load average. */
int
thread_get_load_avg (void) {
    /* TODO: Your implementation goes here */
    enum intr_level old_level = intr_disable();
    int load_ave_value = fp_to_int_round (mult_mixed (load_avg, 100));
    intr_set_level(old_level);
    return load_ave_value;
}

/* Returns 100 times the current thread's recent_cpu value. */
int
thread_get_recent_cpu (void) {
    /* TODO: Your implementation goes here */
    enum intr_level old_level = intr_disable();
    int recent_cpu = fp_to_int_round (mult_mixed (thread_current() -> recent_cpu, 100));
    intr_set_level(old_level);
    return recent_cpu;
}

/* Idle thread.  Executes when no other thread is ready to run.

   The idle thread is initially put on the ready list by
   thread_start().  It will be scheduled once initially, at which
   point it initializes idle_thread, "up"s the semaphore passed
   to it to enable thread_start() to continue, and immediately
   blocks.  After that, the idle thread never appears in the
   ready list.  It is returned by next_thread_to_run() as a
   special case when the ready list is empty. */
static void
idle (void *idle_started_ UNUSED) {
    struct semaphore *idle_started = idle_started_;

    idle_thread = thread_current ();
    sema_up (idle_started);

    for (;;) {
        /* Let someone else run. */
        intr_disable ();
        thread_block ();

        /* Re-enable interrupts and wait for the next one.

           The `sti' instruction disables interrupts until the
           completion of the next instruction, so these two
           instructions are executed atomically.  This atomicity is
           important; otherwise, an interrupt could be handled
           between re-enabling interrupts and waiting for the next
           one to occur, wasting as much as one clock tick worth of
           time.

           See [IA32-v2a] "HLT", [IA32-v2b] "STI", and [IA32-v3a]
           7.11.1 "HLT Instruction". */
        asm volatile ("sti; hlt" : : : "memory");
    }
}

/* Function used as the basis for a kernel thread. */
static void
kernel_thread (thread_func *function, void *aux) {
    ASSERT (function != NULL);

    intr_enable ();       /* The scheduler runs with interrupts off. */
    function (aux);       /* Execute the thread function. */
    thread_exit ();       /* If function() returns, kill the thread. */
}


/* Does basic initialization of T as a blocked thread named
   NAME. */
static void
init_thread (struct thread *t, const char *name, int priority) {
    ASSERT (t != NULL);
    ASSERT (PRI_MIN <= priority && priority <= PRI_MAX);
    ASSERT (name != NULL);

    memset (t, 0, sizeof *t);
    t->status = THREAD_BLOCKED;
    strlcpy (t->name, name, sizeof t->name);
    t->tf.rsp = (uint64_t) t + PGSIZE - sizeof (void *);
    t->priority = priority;
    t->magic = THREAD_MAGIC;

    list_push_back(&all_list, &t->allelem);     //새로 생성한 thread를 all_list에 넣는다. advanced
    t->init_priority = priority;
    t->wait_on_lock = NULL;
    list_init(&t->donations);

    /* advanced */
    t->nice = NICE_DEFAULT;
    t->recent_cpu = RECENT_CPU_DEFAULT;

    /* 자식 리스트 및 세마포어 초기화 */
    list_init(&t->child_list);
    sema_init(&t->wait_sema,0);
    sema_init(&t->fork_sema,0);
    sema_init(&t->free_sema,0);

    t->running = NULL;

}

/* Chooses and returns the next thread to be scheduled.  Should
   return a thread from the run queue, unless the run queue is
   empty.  (If the running thread can continue running, then it
   will be in the run queue.)  If the run queue is empty, return
   idle_thread. */
static struct thread *
next_thread_to_run (void) {
    if (list_empty (&ready_list))
        return idle_thread;
    else
        return list_entry (list_pop_front (&ready_list), struct thread, elem);
}

/* Use iretq to launch the thread */
void
do_iret (struct intr_frame *tf) {
    __asm __volatile(
            "movq %0, %%rsp\n"
            "movq 0(%%rsp),%%r15\n"
            "movq 8(%%rsp),%%r14\n"
            "movq 16(%%rsp),%%r13\n"
            "movq 24(%%rsp),%%r12\n"
            "movq 32(%%rsp),%%r11\n"
            "movq 40(%%rsp),%%r10\n"
            "movq 48(%%rsp),%%r9\n"
            "movq 56(%%rsp),%%r8\n"
            "movq 64(%%rsp),%%rsi\n"
            "movq 72(%%rsp),%%rdi\n"
            "movq 80(%%rsp),%%rbp\n"
            "movq 88(%%rsp),%%rdx\n"
            "movq 96(%%rsp),%%rcx\n"
            "movq 104(%%rsp),%%rbx\n"
            "movq 112(%%rsp),%%rax\n"
            "addq $120,%%rsp\n"
            "movw 8(%%rsp),%%ds\n"
            "movw (%%rsp),%%es\n"
            "addq $32, %%rsp\n"
            "iretq"
            : : "g" ((uint64_t) tf) : "memory");
}

/* Switching the thread by activating the new thread's page
   tables, and, if the previous thread is dying, destroying it.

   At this function's invocation, we just switched from thread
   PREV, the new thread is already running, and interrupts are
   still disabled.

   It's not safe to call printf() until the thread switch is
   complete.  In practice that means that printf()s should be
   added at the end of the function. */
static void
thread_launch (struct thread *th) {
    uint64_t tf_cur = (uint64_t) &running_thread ()->tf;
    uint64_t tf = (uint64_t) &th->tf;
    ASSERT (intr_get_level () == INTR_OFF);

    /* The main switching logic.
     * We first restore the whole execution context into the intr_frame
     * and then switching to the next thread by calling do_iret.
     * Note that, we SHOULD NOT use any stack from here
     * until switching is done. */
    __asm __volatile (
            /* Store registers that will be used. */
            "push %%rax\n"
            "push %%rbx\n"
            "push %%rcx\n"
            /* Fetch input once */
            "movq %0, %%rax\n"
            "movq %1, %%rcx\n"
            "movq %%r15, 0(%%rax)\n"
            "movq %%r14, 8(%%rax)\n"
            "movq %%r13, 16(%%rax)\n"
            "movq %%r12, 24(%%rax)\n"
            "movq %%r11, 32(%%rax)\n"
            "movq %%r10, 40(%%rax)\n"
            "movq %%r9, 48(%%rax)\n"
            "movq %%r8, 56(%%rax)\n"
            "movq %%rsi, 64(%%rax)\n"
            "movq %%rdi, 72(%%rax)\n"
            "movq %%rbp, 80(%%rax)\n"
            "movq %%rdx, 88(%%rax)\n"
            "pop %%rbx\n"              // Saved rcx
            "movq %%rbx, 96(%%rax)\n"
            "pop %%rbx\n"              // Saved rbx
            "movq %%rbx, 104(%%rax)\n"
            "pop %%rbx\n"              // Saved rax
            "movq %%rbx, 112(%%rax)\n"
            "addq $120, %%rax\n"
            "movw %%es, (%%rax)\n"
            "movw %%ds, 8(%%rax)\n"
            "addq $32, %%rax\n"
            "call __next\n"         // read the current rip.
            "__next:\n"
            "pop %%rbx\n"
            "addq $(out_iret -  __next), %%rbx\n"
            "movq %%rbx, 0(%%rax)\n" // rip
            "movw %%cs, 8(%%rax)\n"  // cs
            "pushfq\n"
            "popq %%rbx\n"
            "mov %%rbx, 16(%%rax)\n" // eflags
            "mov %%rsp, 24(%%rax)\n" // rsp
            "movw %%ss, 32(%%rax)\n"
            "mov %%rcx, %%rdi\n"
            "call do_iret\n"
            "out_iret:\n"
            : : "g"(tf_cur), "g" (tf) : "memory"
            );
}

/* Schedules a new process. At entry, interrupts must be off.
 * This function modify current thread's status to status and then
 * finds another thread to run and switches to it.
 * It's not safe to call printf() in the schedule(). */
static void
do_schedule(int status) {
    ASSERT (intr_get_level () == INTR_OFF);
    ASSERT (thread_current()->status == THREAD_RUNNING);
    while (!list_empty (&destruction_req)) {
        struct thread *victim =
            list_entry (list_pop_front (&destruction_req), struct thread, elem);
        palloc_free_page(victim);
    }
    thread_current ()->status = status;
    schedule ();
}

static void
schedule (void) {
    struct thread *curr = running_thread ();
    struct thread *next = next_thread_to_run ();

    ASSERT (intr_get_level () == INTR_OFF);
    ASSERT (curr->status != THREAD_RUNNING);
    ASSERT (is_thread (next));
    /* Mark us as running. */
    next->status = THREAD_RUNNING;

    /* Start new time slice. */
    thread_ticks = 0;

#ifdef USERPROG
    /* Activate the new address space. */
    process_activate (next);
#endif

    if (curr != next) {
        /* If the thread we switched from is dying, destroy its struct
           thread. This must happen late so that thread_exit() doesn't
           pull out the rug under itself.
           We just queuing the page free reqeust here because the page is
           currently used bye the stack.
           The real destruction logic will be called at the beginning of the
           schedule(). */
        if (curr && curr->status == THREAD_DYING && curr != initial_thread) {
            ASSERT (curr != next);
            list_push_back (&destruction_req, &curr->elem);
        }

        /* Before switching the thread, we first save the information
         * of current running. */
        thread_launch (next);
    }
}

/* Returns a tid to use for a new thread. */
static tid_t
allocate_tid (void) {
    static tid_t next_tid = 1;
    tid_t tid;

    lock_acquire (&tid_lock);
    tid = next_tid++;
    lock_release (&tid_lock);

    return tid;
}
/* 가장 먼저 일어나야할 스레드를 비교하여 새로운 값이 작을 경우 변경*/
void 
update_next_tick_to_awake(int64_t ticks){
    /* next_tick_to_awake 가 깨워야할 스레드의 깨어날 tick값 중 가장 작은 tick을 갖도록 업데이트 한다. */
    next_tick_to_awake = (ticks < next_tick_to_awake) ? ticks : next_tick_to_awake;
}

/* 가장 먼저 일어나야할 스레드가 일어날 시각을 반환*/
int64_t
get_next_tick_to_awake(void){
    return next_tick_to_awake;
}

/* 스레드를 ticks시각 까지 재우는 함수*/
void thread_sleep(int64_t ticks){
    
    enum intr_level old_level = intr_disable(); //이전 인터럽트 레벨을 저장하고 인터럽트 방지
    
    struct thread *cur = thread_current(); // idle 스레드는 sleep 되지 않아야한다.
    ASSERT(cur != idle_thread);

    /* 현재 스레드를 슬립 큐에 삽입한 후에 스케줄한다. */
    list_push_back(&sleep_list, &cur->elem);

    //awake함수가 실행되어야 할 tick값을 update
    cur->wakeup_tick = ticks;
    update_next_tick_to_awake(ticks);

    /* 이 스레드를 블락하고 다시 스케줄될 때 까지 블락된 상태로 대기*/
    thread_block();

    /* 인터럽트를 다시 받아들이도록 수정 */
    intr_set_level(old_level);
}


/* 자고 있는 스레드 중에 깨어날 시각이 ticks 시각이 지난 애들을 모조리 깨우는 하수 */
void thread_awake(int64_t ticks){
    next_tick_to_awake = INT64_MAX;
    struct list_elem *e = list_begin(&sleep_list);

    while(e != list_end(&sleep_list))
    {
        struct thread *t = list_entry(e, struct thread, elem);
        if(ticks >= t->wakeup_tick)
        {
            e = list_remove(&t->elem);
            thread_unblock(t);
        }
        else
        {
            e = list_next(e);
            update_next_tick_to_awake(t->wakeup_tick);
        }
    }
}

bool
thread_compare_priority(const struct list_elem *a, const struct list_elem *b, void *aux UNUSED)
{
    return list_entry(a, struct thread, elem) ->priority > list_entry(b, struct thread, elem) -> priority;

}

// 첫번째 스레드가 cpu 점유 중인 스레드 보다 우선순위가 높으면 cpu 점유를 양보하는 함수
void 
test_max_priority(void){
    struct thread *cur = thread_current();
    if (!list_empty(&ready_list) && thread_compare_priority(list_front(&ready_list), cur, 0)){
        thread_yield();
    }
}

bool check_preemption(){
    if(list_empty(&ready_list)) return false;
    return list_entry(list_front(&ready_list), struct thread, elem) -> priority > thread_current() -> priority;
}


bool thread_compare_donate_priority(const struct list_elem *l, const struct list_elem *s, void *aux UNUSED){
    return list_entry(l, struct thread, donation_elem)->priority > list_entry(s, struct thread, donation_elem)->priority;
}

/* 자신의 priority를 필요한 lock을 점유하고 있는 thread에게 빌려주는 함수 */
void donate_priority(){
    int depth;
    struct thread *cur = thread_current();
    for(depth = 0; depth < 8; depth++){
        if(!cur->wait_on_lock) break;
        struct thread *holder = cur->wait_on_lock->holder;
        holder->priority = cur->priority;
        cur = holder;
    }
}

/* 자신에게 priority를 빌려준 thread들을 donations 리스트에서 지우는 함수 */
void remove_with_lock(struct lock *lock){
    struct list_elem *e;
    struct thread *cur = thread_current();

	for (e = list_begin(&cur->donations); e!= list_end(&cur->donations); e = list_next(e)){
		struct thread *t = list_entry(e, struct thread, donation_elem);
		if(t->wait_on_lock == lock){
			list_remove(&t->donation_elem);
		}
	}
}

/* priority를 재설정 */
void refresh_priority(){
    struct thread *cur = thread_current();
    cur->priority = cur->init_priority;
   /* cur->donation에 thread가 남아있다면, 
    * 그 안의 thread들을 priority에 따라 정렬한 후에 높은 우선순위(dontation list 가장 앞에 있는 thread의 priority)를 cur thread의 priority로 설정한다.
    */
    if(!list_empty(&cur->donations)){
        list_sort(&cur->donations, thread_compare_donate_priority, 0);
        struct thread *front = list_entry(list_front(&cur->donations), struct thread, donation_elem);
        cur->priority = front->priority > cur->priority ? front->priority: cur->priority; 
    }
}

/* advance */

void mlfqs_calculate_priority(struct thread *t){
    //priority = PRI_MAX - (recent_cpu / 4) - (nice * 2)
    if(t == idle_thread){
        return;
    }
    t->priority = fp_to_int (add_mixed (div_mixed (t->recent_cpu, -4), PRI_MAX - t->nice * 2));
}

void mlfqs_calculate_recent_cpu(struct thread *t){
    
    if(t == idle_thread){
        return;
    }
    // recent_cpu = (2 * load_avg) / (2 * load_avg + 1) * recent_cpu + nice
    t->recent_cpu = add_mixed (mult_fp (div_fp (mult_mixed (load_avg, 2), 
    add_mixed (mult_mixed (load_avg, 2), 1)), t->recent_cpu), t->nice);
}
void mlfqs_calculate_load_avg(void){

    int ready_threads;

    if (thread_current () == idle_thread)
        ready_threads = list_size (&ready_list);
    else
        ready_threads = list_size (&ready_list) + 1;
    //load_avg = (59/60) * load_avg + (1/60) * ready_threads
    load_avg = add_fp (mult_fp (div_fp (int_to_fp(59), int_to_fp(60)), load_avg), 
                    mult_mixed (div_fp (int_to_fp(1), int_to_fp(60)), ready_threads));
}
/* tick마다 current cpu의 recent_cpu를 1 증가 시킨다. */
void mlfqs_increments_recent_cpu(void){
    if(thread_current() != idle_thread){
        thread_current()->recent_cpu = add_mixed (thread_current()->recent_cpu, 1);
    }
}
/* 모든 list를 돌면서 recent_cpu를 변경한다.*/
void mlfqs_recalculate_recent_cpu(void){
    struct list_elem *e;
    for(e = list_begin(&all_list) ; e != list_end(&all_list); e = list_next(e)){
        struct thread *t = list_entry(e, struct thread, allelem);
        mlfqs_calculate_recent_cpu(t);
    }
}

/* 모든 list를 돌면서 priority를 변경시켜준다.*/
void mlfqs_recalculate_priority(void){
    struct list_elem *e;
    for(e = list_begin(&all_list) ; e != list_end(&all_list); e = list_next(e)){
        struct thread *t = list_entry(e, struct thread, allelem);
        mlfqs_calculate_priority(t);
    }
}


/***********************************************/
/*           Fixed-point arithmetic            */
/***********************************************/
/* 
    pintos에서는 1bit(부호), 17bit(정수부), 14bit(소수부)를 사용하는 고정 소수점 방식으로 실수를 계산한다.
    그렇기 때문에, 정수와 실수의 계산을 할때에는 다른 연산 방법을 사용해야 한다.
    x,y는 fixed_point num
    n은 int
    F는 1<<14
*/

int int_to_fp (int n) {
  return n * F;
}

int fp_to_int (int x) {
  return x / F;
}

int fp_to_int_round (int x) {
  if (x >= 0) return (x + F / 2) / F;
  else return (x - F / 2) / F;
}

int add_fp (int x, int y) {
  return x + y;
}

int sub_fp (int x, int y) {
  return x - y;
}

int add_mixed (int x, int n) {
  return x + n * F;
}

int sub_mixed (int x, int n) {
  return x - n * F;
}

int mult_fp (int x, int y) {
  return ((int64_t) x) * y / F;
}

int mult_mixed (int x, int n) {
  return x * n;
}

int div_fp (int x, int y) {
  return ((int64_t) x) * F / y;
}

int div_mixed (int x, int n) {
  return x / n;
}