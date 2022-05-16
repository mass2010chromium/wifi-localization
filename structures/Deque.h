#pragma once
#include <stdbool.h>
#include <stddef.h>

/**
 * Fixed size Deque data structure, for undo.
 */

struct Deque {
    size_t head;
    size_t tail;   // Past-end: insert at tail index
    size_t size;
    size_t max_size;
    void** elements;
};

typedef struct Deque Deque;

/**
 * Make an empty deque with a specified max size.
 */
Deque* make_Deque(size_t size);

void inplace_make_Deque(Deque* deque, size_t size);

size_t Deque_size(Deque* this);

bool Deque_full(Deque* this);

bool Deque_empty(Deque* this);

/**
 * Resize the Deque. WARNING: CAN DELETE ELEMENTS IMPLICITLY!
 */
void Deque_resize(Deque* this, size_t new_capacity);

/**
 * Push onto the end of the deque.
 */
int Deque_push(Deque* this, void* element);

/**
 * Push onto the front
 */
int Deque_push_l(Deque* this, void* element);

/**
 * Pop from the front of the deque.
 */
void* Deque_pop(Deque* this);

/**
 * Pop from the end of the deque.
 */
void* Deque_pop_r(Deque* this);

/**
 * Get right side element without removing it.
 */
void** Deque_peek_r(Deque* this);

/**
 * Get left side element without removing it.
 */
void** Deque_peek_l(Deque* this);

void Deque_destroy(Deque* v);
