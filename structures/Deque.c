#include "Deque.h"

#include <assert.h>
#include <stdlib.h>

/**
 * Make an empty deque with a specified max size.
 */
Deque* make_deque(size_t size) {
    //TODO error checking
    Deque* ret = malloc(sizeof(Deque));
    inplace_make_Deque(ret, size);
    return ret;
}

void inplace_make_Deque(Deque* deque, size_t size) {
    deque->head = 0;
    deque->tail = 0;
    deque->size = 0;
    deque->max_size = size;
    deque->elements = malloc(size * sizeof(void*));
}

size_t Deque_size(Deque* this) {
    return this->size;
}

bool Deque_full(Deque* this) {
    return this->size == this->max_size;
}

bool Deque_empty(Deque* this) {
    return this->size == 0;
}

/**
 * Resize the Deque. WARNING: CAN DELETE ELEMENTS IMPLICITLY!
 */
void Deque_resize(Deque* this, size_t new_capacity) {
    void** new_elements = malloc(new_capacity * sizeof(void*));
    size_t keep_count = this->size;
    if (keep_count > new_capacity) { keep_count = new_capacity; }

    size_t index = this->head;
    for (size_t i = 0; i < keep_count; ++i) {
        new_elements[i] = this->elements[index];
        ++index;
        if (index == this->max_size) { index = 0; }
    }
    free(this->elements);
    this->size = keep_count;
    this->max_size = new_capacity;
    this->head = 0;
    this->tail = keep_count;
    if (this->tail == new_capacity) { this->tail = 0; }
}

/**
 * Push onto the end of the deque.
 */
int Deque_push(Deque* this, void* element) {
    if (Deque_full(this)) {
        return -1;
    }
    this->elements[this->tail] = element;
    this->tail = (this->tail + 1) % this->max_size;
    ++this->size;
    return 0;
}

/**
 * Push onto the front.
 */
int Deque_push_l(Deque* this, void* element) {
    if (Deque_full(this)) {
        return -1;
    }
    if (this->head == 0) { this->head = this->max_size - 1; }
    else { --this->head; }
    this->elements[this->head] = element;
    ++this->size;
    return 0;
}

/**
 * Pop from the front of the deque.
 */
void* Deque_pop(Deque* this) {
    assert(this->size > 0);
    if (Deque_empty(this)) {
        //TODO error handling?
        return NULL;
    }
    void* elt = this->elements[this->head];
    this->head = (this->head + 1) % this->max_size;
    --this->size;
    return elt;
}

/**
 * Pop from the end of the deque.
 */
void* Deque_pop_r(Deque* this) {
    assert(this->size > 0);
    if (Deque_empty(this)) {
        //TODO error handling?
        return NULL;
    }
    if (this->tail == 0) { this->tail = this->max_size - 1; }
    else { --this->tail; }
    void* elt = this->elements[this->tail];
    --this->size;
    return elt;
}

/**
 * Get right side element without removing it.
 */
void** Deque_peek_r(Deque* this) {
    assert(this->size > 0);
    size_t last_idx;
    if (this->tail == 0) { last_idx = this->max_size - 1; }
    else { last_idx = this->tail - 1; }
    return &this->elements[last_idx];
}

/**
 * Get left side element without removing it.
 */
void** Deque_peek_l(Deque* this) {
    assert(this->size > 0);
    return &this->elements[this->head];
}

void Deque_destroy(Deque* this) {
    free(this->elements);
    this->elements = NULL;
}
