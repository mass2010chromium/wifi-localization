#pragma once
#include <stddef.h>

struct String {
    size_t length;
    size_t max_length;
    char data[0];
};

typedef struct String String;

/**
 * Allocates space for a new String object,
 * initializes it using alloc_string, then
 * copies the contents of `data` to the new String
 * and returns it.
 */
String* make_String(const char* data);

/**
 * Take a malloc'd string and realloc it into a String.
 */
String* convert_String(char* data);
/**
 * Take a malloc'd String and "realloc" it into a string.
 */
char* String_to_cstr(String* data);

/**
 * Allocates space for a new String object on the heap
 * then initializes it with default values.
 */
String* alloc_String(size_t);

#define static_String(name, init_sz) \
static String* name = NULL; \
if (name == NULL) { name = alloc_String(init_sz); } \
else { String_clear(name); }

/**
 * Reallocates the space allocated for some String* to
 * accomodate some new size_t, if needed. Also updates the max_length
 * of the passed String* if its storage space was reallocated.
 */
String* realloc_String(String*, size_t);

/**
 * Returns s->length.
 */
size_t Strlen(const String* s);

String* Strdup(const String* s);
String* Strndup(const String* s, size_t count);

/**
 * Make substring. (allocates on heap)
 */
String* Strsub(const String* s, size_t start, size_t end);

/*
 * Might modify the first pointer (length extend).
 * Append string b to string a.
 * Also returns the new pointer if u want to use that instead.
 */
String* Strcat(String**, const String*);

/**
 * @see Strcat, except second argument is a C string
 */
String* Strcats(String**, const char*);

/**
 * @see Strcats, except only n bytes are copied.
 */
String* Strncats(String**, const char*, size_t);

/**
 * Set the content of this string.
 * May realloc.
 */
void Strcpy(String**, String*);
void Strncpys(String**, char*, size_t);
void Strcpys(String**, char*);

/**
 * Push a char onto the end of this string. Increases length.
 * Might cause a reallocation.
 */
void String_push(String**, char);

/**
 * Removes a char from the end of this string and returns it.
 * Decreases length. Will not realloc.
 */
char String_pop(String*);

/**
 * Removes all chars from this string.
 * Decreases length. Will not realloc.
 * To free memory you have to free() the String.
 */
void String_clear(String*);

/**
 * Delete character at index given (0-indexed) and returns it.
 * Does not resize (maxlen) string.
 */
char String_delete(String* s, size_t index);

/**
 * Delete characters [a, b).
 * Does not resize (maxlen) string.
 */
void String_delete_range(String* s, size_t a, size_t b);

/**
 * Truncate string.
 */
void Strtrunc(String* s, size_t length);

/**
 * Postcondition: (*s)->data[index] == c
 */
void String_insert(String** s, size_t index, char c);

/**
 * Postcondition: (*s)->data[index:index+strlen(insert)] == insert
 */
void String_inserts(String** s, size_t index, const char* insert);

/**
 * Postcondition: (*s)->data[index:index+n] == insert[:n]
 */
void String_ninserts(String** s, size_t index, const char* insert, size_t n);

/**
 * "Fit" the string (realloc to fit length).
 * Use:
 *      String* s = ...;
 *      s = String_fit(s);
 */
String* String_fit(String* s);
