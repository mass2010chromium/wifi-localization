#pragma once

#include <stddef.h>

struct Vector {
    size_t size;
    size_t max_size;
    void** elements;
};

typedef struct Vector Vector;

/**
 * Allocates space for a newly created Vector,
 * initializes it using inplace_make_Vector, then returns it.
 */
Vector* make_Vector(size_t init_size);

/**
 * Sets the size of `vector` to 0, max_size to init_size, and
 * allocates `init_size` bytes of memory for `vector->elements`.
 */
void inplace_make_Vector(Vector* vector, size_t init_size);

/**
 * Vector shallow copy.
 */
Vector* Vector_copy(Vector* v);

/**
 * Vector push onto end.
 */
void Vector_push(Vector* v, void* element);

/**
 * Vector pop from end. (O(1) end)
 */
void* Vector_pop(Vector* v);

/**
 * Vector insert into middle.
 * Postcondition: v[idx] = element
 */
void Vector_insert(Vector* v, size_t idx, void* element);

/**
 * Vector allocate a block of space in the middle of the vector.
 * Allocated space is not initialized (contains hot garbage).
 * Postcondition: v[idx ... idx+count-1] = undefined
 */
void Vector_create_range(Vector* v, size_t idx, size_t count);

/**
 * Vector delete element at index. Shifts everything past it left.
 */
void Vector_delete(Vector* v, size_t idx);

/**
 * Vector delete element in range [a, b). Shift everything >=b left.
 * Zero length ranges allowed (as long as b <= v.size).
 */
void Vector_delete_range(Vector* v, size_t a, size_t b);

/**
 * Quicksort. Works (probably).
 * Not wrapping qsort for less pointery-ness.
 * Compare function: cmp(a, b) returns "a - b".
 */
void Vector_sort(Vector* v, int(*cmp)(void*, void*));

/**
 * Resets the size of the vector to 0, max_size to init_size,
 * and reallocs the vector contents to fit init_size elements.
 * NOTE: Doesn't call free() on any of the contained elements!
 */
void Vector_clear(Vector* v, size_t init_size);

/**
 * This one frees its contained elements.
 */
void Vector_clear_free(Vector* v, size_t init_size);

/**
 * Frees v->contents and sets the pointer to NULL.
 * NOTE: Doesn't call free() on the passed in pointer!
 */
void Vector_destroy(Vector* v);

/**
 * Casts the arguments to ssize_t and returns if arg1 > arg2.
 */
int signed_compare(void*, void*);

/**
 * Casts the arguments to size_t and returns if arg1 > arg2.
 */
int unsigned_compare(void*, void*);

