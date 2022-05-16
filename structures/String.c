#include "String.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

String* make_String(const char* data) {
    String* ret = alloc_String(strlen(data));
    ret->length = strlen(data);
    memcpy(ret->data, data, strlen(data)+1);
    return ret;
}

/**
 * Take a malloc'd string and realloc it into a String.
 */
String* convert_String(char* data) {
    size_t maxlen = strlen(data);
    String* ret = realloc(data, sizeof(String) + maxlen + 1);
    memmove(ret + 1, ret, maxlen+1);
    ret->length = maxlen;
    ret->max_length = maxlen;
    return ret;
}

/**
 * Take a malloc'd String and "realloc" it into a string.
 */
char* String_to_cstr(String* data) {
    char* ret = (char*) data;
    memmove(ret, data+1, data->length+1);
    return ret;
}

String* alloc_String(size_t maxlen) {
    String* ret = malloc(sizeof(String) + maxlen + 1);
    ret->data[0] = 0;
    ret->length = 0;
    ret->max_length = maxlen;
    return ret;
}

String* realloc_String(String* s, size_t maxlen) {
    if (s->max_length > maxlen) {
        return s;
    }
    if (maxlen < 2 * s->max_length && maxlen < 4096) {
        maxlen = 2 * s->max_length;
    }
    String* ret = (String*) realloc(s, sizeof(String) + maxlen + 1);
    ret->max_length = maxlen;
    if (ret->length > maxlen) {
        ret->length = maxlen;
        ret->data[maxlen] = 0;
    }
    return ret;
}

size_t Strlen(const String* s) {
    return s->length;
}

String* Strdup(const String* s) {
    size_t data_size = sizeof(String) + s->max_length + 1;
    String* ret = malloc(data_size);
    memcpy(ret, s, data_size);
    return ret;
}

String* Strndup(const String* s, size_t count) {
    assert(count <= s->length);
    String* ret = alloc_String(count);
    memcpy(ret->data, s->data, count+1);
    ret->length = count;
    return ret;
}

/**
 * Make substring. (allocates on heap)
 */
String* Strsub(const String* s, size_t start, size_t end) {
    assert(end >= start);
    assert(start < s->length);
    size_t count = end - start;
    String* ret = alloc_String(count);
    memcpy(ret->data, s->data + start, count);
    ret->length = count;
    ret->data[ret->length] = 0;
    return ret;
}

/*
 * Might modify the first pointer (length extend).
 * Append string b to string a.
 * Also returns the new pointer if u want to use that instead.
 */
String* Strcat(String** _a, const String* b) {
    String* a = *_a;
    size_t blen = Strlen(b);
    String* ret = realloc_String(a, Strlen(a) + blen);
    memcpy(ret->data + Strlen(ret), b->data, blen+1);
    ret->length += blen;
    *_a = ret;
    return ret;
}

/**
 * @see Strcat, except second argument is a C string
 */
String* Strcats(String** _a, const char* b) {
    return Strncats(_a, b, strlen(b));
}

/**
 * @see Strcats, except only n bytes are copied.
 */
String* Strncats(String** _a, const char* b, size_t blen) {
    String* a = *_a;
    String* ret = realloc_String(a, Strlen(a) + blen);
    memcpy(ret->data + Strlen(ret), b, blen);
    ret->length += blen;
    ret->data[ret->length] = 0;
    *_a = ret;
    return ret;
}

void Strcpy(String** _s, String* src) {
    Strncpys(_s, src->data, src->length);
}

void Strncpys(String** _s, char* dat, size_t length) {
    String* s = *_s;
    if (s->length < length) {
        free(s);
        *_s = make_String(dat);
    }
    else {
        strncpy(s->data, dat, length+1);
        s->length = length;
    }
}

void Strcpys(String** _s, char* dat) {
    Strncpys(_s, dat, strlen(dat));
}


void String_push(String** _s, char c) {
    String* s = *_s;
    if (s->length < s->max_length) {
        s->data[s->length] = c;
        s->data[s->length+1] = 0;
        ++s->length;
    }
    else {
        String* new = realloc_String(s, s->max_length * 2 + 1);
        String_push(&new, c);
        *_s = new;
    }
}

char String_pop(String* s) {
    s->length -= 1;
    char ret = s->data[s->length];
    s->data[s->length] = 0;
    return ret;
}

void String_clear(String* s) {
    s->length = 0;
    s->data[0] = 0;
}

/**
 * Delete character at index given (0-indexed) and returns it.
 * Does not resize (maxlen) string.
 */
char String_delete(String* s, size_t index) {
    size_t line_len = s->length;
    size_t rest = line_len - (index + 1);
    char ret = s->data[index];
    s->length -= 1;
    memmove(s->data + index, s->data + index+1, rest+1);
    return ret;
}

/**
 * Delete characters [a, b).
 * Does not resize (maxlen) string.
 */
void String_delete_range(String* s, size_t a, size_t b) {
    assert(b >= a);
    assert(b <= s->length);
    size_t line_len = s->length;
    size_t rest = line_len - b;
    s->length -= (b-a);
    memmove(s->data + a, s->data + b, rest+1);
}

/**
 * Truncate string.
 */
void Strtrunc(String* s, size_t length) {
    assert(length <= s->length);
    s->data[length] = 0;
    s->length = length;
}

/**
 * Postcondition: (*s)->data[index] == c
 */
void String_insert(String** _s, size_t index, char c) {
    String* s = *_s;
    if (s->length == s->max_length) {
        s = realloc_String(s, s->max_length + 1);
        *_s = s;
    }
    size_t tail = s->length - index;
    s->length += 1;
    memmove(s->data+index+1, s->data+index, tail+1);
    s->data[index] = c;
}

/**
 * Postcondition: (*s)->data[index:index+strlen(insert)] == insert
 */
void String_inserts(String** s, size_t index, const char* insert) {
    String_ninserts(s, index, insert, strlen(insert));
}

/**
 * Postcondition: (*s)->data[index:index+n] == insert[:n]
 */
void String_ninserts(String** _s, size_t index, const char* insert, size_t n) {
    String* s = *_s;
    size_t new_length = s->length + n;
    if (new_length > s->max_length) {
        s = realloc_String(s, new_length);
        *_s = s;
    }
    size_t tail = s->length - index;
    s->length += n;
    memmove(s->data+index+n, s->data+index, tail+1);
    memcpy(s->data + index, insert, n);
}

/**
 * "Fit" the string (realloc to fit length).
 * Use:
 *      String* s = ...;
 *      s = String_fit(s);
 */
String* String_fit(String* s) {
    String* ret = (String*) realloc(s, sizeof(String) + s->length + 1);
    ret->max_length = ret->length;
    return ret;
}
