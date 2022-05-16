#include "Vector.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

Vector* make_Vector(size_t init_size) {
    //TODO error checking
    Vector* ret = malloc(sizeof(Vector));
    inplace_make_Vector(ret, init_size);
    return ret;
}

void inplace_make_Vector(Vector* vector, size_t init_size) {
    vector->size = 0;
    vector->max_size = init_size;
    vector->elements = malloc(init_size * sizeof(void*));
}

/**
 * Vector shallow copy.
 */
Vector* Vector_copy(Vector* v) {
    // TODO error handling
    Vector* ret = malloc(sizeof(Vector));
    ret->size = v->size;
    ret->max_size = v->max_size;
    ret->elements = malloc(v->max_size * sizeof(void*));
    memcpy(ret->elements, v->elements, v->size * sizeof(void*));
    return ret;
}

/**
 * Vector push onto end.
 */
void Vector_push(Vector* v, void* element) {
    if (v->size == v->max_size) {
        // TODO error handling
        v->elements = realloc(v->elements, v->max_size*2 * sizeof(void*));
        v->max_size *= 2;
    }
    v->elements[v->size] = element;
    v->size += 1;
}

/**
 * Vector pop from end. (O(1) end)
 */
void* Vector_pop(Vector* v) {
    assert(v->size > 0);
    void* ret = v->elements[v->size-1];
    --v->size;
    return ret;
}

/**
 * Vector insert into middle.
 * Postcondition: v[idx] = element
 */
void Vector_insert(Vector* v, size_t idx, void* element) {
    if (v->size == v->max_size) {
        // TODO error handling
        v->elements = realloc(v->elements, v->max_size*2 * sizeof(void*));
        v->max_size *= 2;
    }
    memmove(v->elements + idx + 1, v->elements + idx, (v->size - idx) * sizeof(void*));
    v->elements[idx] = element;
    v->size += 1;
}

/**
 * Vector allocate a block of space in the middle of the vector.
 * Allocated space is not initialized (contains hot garbage).
 * Postcondition: v[idx ... idx+count-1] = undefined
 */
void Vector_create_range(Vector* v, size_t idx, size_t count) {
    const size_t new_size = v->size + count;
    if (new_size > v->max_size) {
        // TODO error handling
        size_t target_size = v->max_size * 2;
        if (target_size < new_size) {
            target_size = new_size;
        }
        v->elements = realloc(v->elements, target_size * sizeof(void*));
        v->max_size = target_size;
    }
    memmove(v->elements + idx + count, v->elements + idx, (v->size - idx) * sizeof(void*));
    v->size = new_size;
}

/**
 * Vector delete element at index. Shifts everything past it left.
 */
void Vector_delete(Vector* v, size_t idx) {
    memmove(v->elements + idx, v->elements + idx+1, (v->size - (idx + 1)) * sizeof(void*));
    v->size -= 1;
}

/**
 * Vector delete element in range [a, b). Shift everything >=b left.
 * Zero length ranges allowed (as long as b <= v.size).
 */
void Vector_delete_range(Vector* v, size_t a, size_t b) {
    assert(b >= a);
    assert(v->size >= b);
    memmove(v->elements + a, v->elements + b, (v->size - b) * sizeof(void*));
    v->size -= b-a;
}

void _Vector_quicksort(Vector* v, size_t lower, size_t upper, int(*cmp)(void*, void*)) {
    if (upper - lower <= 1) return;
    void** arr = v->elements;
    // Improve performance on sorted lists
    size_t pivot_idx = (upper + lower) / 2;
    void* pivot = arr[pivot_idx];
    arr[pivot_idx] = arr[lower];
    size_t start = lower + 1;
    size_t end = upper;
    while (start != end) {
        if (cmp(arr[start], pivot) > 0) {
            --end;
            void* end_elt = arr[end];
            arr[end] = arr[start];
            arr[start] = end_elt;
        }
        else {
            arr[start-1] = arr[start];
            ++start;
        }
    }
    arr[start-1] = pivot;
    _Vector_quicksort(v, lower, start-1, cmp);
    _Vector_quicksort(v, start, upper, cmp);
}
void Vector_sort(Vector* v, int(*cmp)(void*, void*)) {
    _Vector_quicksort(v, 0, v->size, cmp);
}

int signed_compare(void* a, void* b) {
    return ((ssize_t) a) > ((ssize_t) b);
}
int unsigned_compare(void* a, void* b) {
    return ((size_t) a) > ((size_t) b);
}

/**
 * NOTE: Doesn't call free() on any of the contained elements!
 */
void Vector_clear(Vector* v, size_t init_size) {
    v->size = 0;
    v->max_size = init_size;
    v->elements = realloc(v->elements, init_size * sizeof(void*));
}

/**
 * This one frees its contained elements.
 */
void Vector_clear_free(Vector* v, size_t init_size) {
    for (size_t i = 0; i < v->size; ++i) {
        free(v->elements[i]);
    }
    v->size = 0;
    v->max_size = init_size;
    v->elements = realloc(v->elements, init_size * sizeof(void*));
}

/**
 * NOTE: Doesn't call free() on the passed in pointer!
 */
void Vector_destroy(Vector* this) {
    free(this->elements);
    this->elements = NULL;
}

