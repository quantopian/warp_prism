#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Python.h"
#include "numpy/arrayobject.h"

const char* const signature = "PGCOPY\n\377\r\n\0";
const size_t signature_len = 11;

const size_t column_buffer_growth_factor = 2;
const size_t starting_column_buffer_length = 4096;

#ifdef __ORDER_LITTLE_ENDIAN__
#define MAYBE_BSWAP(arg, size) __builtin_bswap ## size (arg)
#else
#define MAYBE_BSWAP(arg, size) arg
#endif

#ifndef likely
#define likely(p) __builtin_expect(!!(p), 1)
#endif

#ifndef unlikely
#define unlikely(p) __builtin_expect(!!(p), 0)
#endif

#define TYPE(size) uint ## size ## _t

#define DEFINE_READ(size)                                               \
    static inline TYPE(size) read ## size (const char* buffer) {        \
        return MAYBE_BSWAP(*(TYPE(size)*) buffer, size);                \
    }

static inline uint8_t read8(const char* buffer) {
    return *buffer;
}

DEFINE_READ(16)
DEFINE_READ(32)
DEFINE_READ(64)

#undef DEFINE_READ

#define DEFINE_WRITE(size)                                              \
    static inline TYPE(size) write ## size (char* buffer, TYPE(size) value) { \
        return *((TYPE(size)*) buffer) = value;                         \
    }

DEFINE_WRITE(8)
DEFINE_WRITE(16)
DEFINE_WRITE(32)
DEFINE_WRITE(64)

#undef TYPE
#undef DEFINE_WRITE

typedef int (*parse_function)(char* column_buffer,
                              const char * const input_buffer,
                              size_t len);
typedef void (*free_function)(void* colbuffer, size_t rowcount);
typedef int (*write_null_function)(char* dst, size_t size);

typedef struct {
    const char* const dtype_name;
    parse_function parse;
    free_function free;
    write_null_function write_null;
    size_t size;
    PyArray_Descr* dtype;
} warp_prism_type;

static int parse_int16(char* column_buffer,
                       const char* const input_buffer,
                       size_t len) {
    if (unlikely(len != sizeof(uint16_t))) {
        PyErr_Format(PyExc_ValueError, "mismatched int16 size: %zu", len);
        return -1;
    }

    write16(column_buffer, read16(input_buffer));
    return 0;
}

static int parse_int32(char* column_buffer,
                       const char* const input_buffer,
                       size_t len) {
    if (unlikely(len != sizeof(uint32_t))) {
        PyErr_Format(PyExc_ValueError, "mismatched int32 size: %zu", len);
        return -1;
    }

    write32(column_buffer, read32(input_buffer));
    return 0;
}

static int parse_int64(char* column_buffer,
                       const char* const input_buffer,
                       size_t len) {
    if (unlikely(len != sizeof(uint64_t))) {
        PyErr_Format(PyExc_ValueError, "mismatched int64 size: %zu", len);
        return -1;
    }

    write64(column_buffer, read64(input_buffer));
    return 0;
}

/* 2000-01-01 in us since 1970-01-01 00:00:00+0000
   postgres stores datetimes as us since jan 1 2000 *not* jan 1 1970 */
const int64_t datetime_offset = 946684800000000l;

static int parse_datetime(char* column_buffer,
                          const char* const input_buffer,
                          size_t len) {
    if (unlikely(len != sizeof(int64_t))) {
        PyErr_Format(PyExc_ValueError, "mismatched datetime size: %zu", len);
        return -1;
    }

    write64(column_buffer, read64(input_buffer) + datetime_offset);
    return 0;
}

/* 2000-01-01 in days since 1970-01-01;
   postgres stores date as days since jan 1 2000 *not* jan 1 1970 */
const int32_t date_offset = 10957;

static int parse_date(char* column_buffer,
                      const char* const input_buffer,
                      size_t len) {
    if (unlikely(len != sizeof(int32_t))) {
        PyErr_Format(PyExc_ValueError, "mismatched date size: %zu", len);
        return -1;
    }

    /* We read 32 bits of data and but write it as 64 bits; postgres uses 32 bit
       integers for dates but numpy datetime64[D] uses 64. */
    write64(column_buffer, read32(input_buffer) + date_offset);
    return 0;
}

static int parse_float32(char* column_buffer,
                         const char* const input_buffer,
                         size_t len) {
    if (unlikely(len != sizeof(float))) {
        PyErr_Format(PyExc_ValueError, "mismatched float32 size: %zu", len);
        return -1;
    }

    write32(column_buffer, read32(input_buffer));
    return 0;
}

static int parse_float64(char* column_buffer,
                         const char* const input_buffer,
                         size_t len) {
    if (unlikely(len != sizeof(double))) {
        PyErr_Format(PyExc_ValueError, "mismatched float64 size: %zu", len);
        return -1;
    }

    write64(column_buffer, read64(input_buffer));
    return 0;
}

static int parse_bool(char* column_buffer,
                      const char* const input_buffer,
                      size_t len) {
    if (unlikely(len != sizeof(uint8_t))) {
        PyErr_Format(PyExc_ValueError, "mismatched bool size: %zu", len);
        return -1;
    }

    write8(column_buffer, read8(input_buffer));
    return 0;
}

static int parse_text(char* column_buffer,
                      const char* const input_buffer,
                      size_t len) {
    PyObject* value = PyUnicode_FromStringAndSize(input_buffer, len);
    if (unlikely(!value)) {
        return -1;
    }

    *(PyObject**) column_buffer = value;
    return 0;
}

static void simple_free(void* colbuffer,
                        size_t rowcount __attribute__((unused))) {
    PyMem_Free(colbuffer);
}

static void free_object(PyObject** colbuffer, size_t rowcount) {
    for (size_t n = 0; n < rowcount; ++n) {
        Py_XDECREF(colbuffer[n]);
    }

    PyMem_Free(colbuffer);
}

static int simple_write_null(char* dst, size_t size) {
    memset(dst, 0, size);
    return 0;
}

static int datetime_write_null(char* dst, size_t size) {
    if (size != sizeof(int64_t)) {
        PyErr_Format(PyExc_ValueError,
                     "wrong size for NULL datetime field: %zu, expected %zu",
                     size,
                     sizeof(int64_t));
        return -1;
    }

    write64(dst, NPY_DATETIME_NAT);
    return 0;
}

static int object_write_null(char* dst, size_t size) {
    if (size != sizeof(PyObject*)) {
        PyErr_Format(PyExc_ValueError,
                     "wrong size for NULL object field: %zu, expected %zu",
                     size,
                     sizeof(PyObject*));
        return -1;
    }

    Py_INCREF(Py_None);
    *(PyObject**) dst = Py_None;
    return 0;
}

warp_prism_type int16_type = {
    "int16",
    (parse_function) parse_int16,
    simple_free,
    simple_write_null,
    sizeof(int16_t),
    NULL,
};

warp_prism_type int32_type = {
    "int32",
    (parse_function) parse_int32,
    simple_free,
    simple_write_null,
    sizeof(uint32_t),
    NULL,
};

warp_prism_type int64_type = {
    "int64",
    (parse_function) parse_int64,
    simple_free,
    simple_write_null,
    sizeof(int64_t),
    NULL,
};

warp_prism_type float32_type = {
    "float32",
    (parse_function) parse_float32,
    simple_free,
    simple_write_null,
    sizeof(float),
    NULL,
};

warp_prism_type float64_type = {
    "float64",
    (parse_function) parse_float64,
    simple_free,
    simple_write_null,
    sizeof(double),
    NULL,
};

warp_prism_type bool_type = {
    "bool",
    (parse_function) parse_bool,
    simple_free,
    simple_write_null,
    sizeof(bool),
    NULL,
};

warp_prism_type string_type = {
    "object",
    (parse_function) parse_text,
    (free_function) free_object,
    object_write_null,
    sizeof(PyObject*),
    NULL,
};

warp_prism_type datetime_type = {
    "datetime64[us]",
    (parse_function) parse_datetime,
    simple_free,
    datetime_write_null,
    sizeof(int64_t),
    NULL,
};

warp_prism_type date_type = {
    "datetime64[D]",
    (parse_function) parse_date,
    simple_free,
    datetime_write_null,
    sizeof(int64_t),
    NULL,
};

const warp_prism_type* typeids[] = {
    &int16_type,
    &int32_type,
    &int64_type,
    &float32_type,
    &float64_type,
    &bool_type,
    &string_type,
    &datetime_type,
    &date_type,
};

const size_t max_typeid = sizeof(typeids) / sizeof(warp_prism_type*);

static inline bool have_oids(uint32_t flags) {
    return flags & (1 << 16);
}

static inline bool valid_flags(uint32_t flags) {
    return flags == 0 || flags == (1 << 16);
}

static inline bool assert_can_consume(size_t size,
                                      size_t cursor,
                                      size_t buffer_len) {
    size_t new_cursor = cursor + size;
    /* unsigned integer overflow is defined to wrap back to 0, if adding size
       to cursor is ever less than cursor then we must have overflowed */
    if (new_cursor < cursor) {
        PyErr_Format(PyExc_ValueError,
                     "consuming %zu bytes would cause an overflow",
                     size);
        return true;
    }
    /* new_cursor is an index into the buffer which must be less than or
       equal to the length of the buffer */
    if (new_cursor > buffer_len) {
        PyErr_Format(PyExc_ValueError,
                     "reading %zu bytes would cause an out of bounds access",
                     size);
        return true;
    }
    return false;
}

static inline uint16_t consume16(const char* buffer, size_t* cursor) {
    uint16_t ret = read16(&buffer[*cursor]);
    *cursor += sizeof(int16_t);
    return ret;
}

static inline bool checked_consume16(const char* buffer,
                                     size_t* cursor,
                                     size_t buffer_len,
                                     uint16_t* out) {
    if (assert_can_consume(sizeof(uint16_t), *cursor, buffer_len)) {
        return true;
    }
    *out = consume16(buffer, cursor);
    return false;
}

static inline uint32_t consume32(const char* buffer, size_t* cursor) {
    uint32_t ret = read32(&buffer[*cursor]);
    *cursor += sizeof(int32_t);
    return ret;
}

static inline bool checked_consume32(const char* buffer,
                                     size_t* cursor,
                                     size_t buffer_len,
                                     uint32_t* out) {
    if (assert_can_consume(sizeof(uint32_t), *cursor, buffer_len)) {
        return true;
    }
    *out = consume32(buffer, cursor);
    return false;
}

static inline void free_outarrays(uint16_t ncolumns,
                                  size_t rowcount,
                                  const warp_prism_type** column_types,
                                  char** outarrays,
                                  bool** outmasks) {
    for (uint_fast16_t n = 0; n < ncolumns; ++n) {
        column_types[n]->free(outarrays[n], rowcount);
        PyMem_Free(outmasks[n]);
    }
}

static inline int allocate_outarrays(uint16_t ncolumns,
                                     const warp_prism_type** column_types,
                                     char** outarrays,
                                     bool** outmasks) {
    uint_fast16_t n;

    for (n = 0; n < ncolumns; ++n) {
        outarrays[n] = PyMem_Malloc(starting_column_buffer_length *
                                    column_types[n]->size);
        if (!outarrays[n]) {
            goto error;
        }

        outmasks[n] = PyMem_Malloc(starting_column_buffer_length *
                                   sizeof(bool));
        if (!outmasks[n]) {
            free(outarrays[n]);
            goto error;
        }
    }
    return 0;

error:
    free_outarrays(n - 1,
                   starting_column_buffer_length,
                   column_types,
                   outarrays,
                   outmasks);
    return -1;
}

static inline int grow_outarrays(uint16_t ncolumns,
                                 size_t* row_count,
                                 const warp_prism_type** column_types,
                                 char** outarrays,
                                 bool** outmasks) {
    size_t new_mask_size;
    uint_fast16_t n;

    *row_count *= column_buffer_growth_factor;
    new_mask_size = *row_count * column_buffer_growth_factor * sizeof(bool);

    for (n = 0; n < ncolumns; ++n) {
        bool* newmask;
        char* new = PyMem_Realloc(outarrays[n],
                                  column_types[n]->size * *row_count);

        if (!new) {
            goto error;
        }
        outarrays[n] = new;

        newmask = PyMem_Realloc(outmasks[n], new_mask_size);
        if (!new) {
            free(outarrays[n]);
            goto error;
        }
        outmasks[n] = newmask;
    }
    return 0;
error:
    free_outarrays(n - 1,
                   *row_count,
                   column_types,
                   outarrays,
                   outmasks);
    return -1;
}

int warp_prism_read_binary_results(const char* const input_buffer,
                                   size_t input_len,
                                   uint16_t const ncolumns,
                                   const warp_prism_type** column_types,
                                   size_t* written_rows,
                                   char** outarrays,
                                   bool** outmasks) {
    size_t cursor = 0;
    uint32_t flags;
    size_t row_count = 0;
    size_t allocated_rows = starting_column_buffer_length;
    uint32_t extension_area;

    if (input_len < signature_len ||
        memcmp(input_buffer, signature, signature_len)) {

        PyErr_SetString(PyExc_ValueError, "missing postgres signature");
        return -1;
    }

    /* advance the cursor through up to the flags segment */
    cursor += signature_len;

    /* flags field */
    if (checked_consume32(input_buffer,
                          &cursor,
                          input_len,
                          &flags)) {
        return -1;
    }

    if (!valid_flags(flags)) {
        PyErr_SetString(PyExc_ValueError, "invalid flags in header");
        return -1;
    }

    /* skip header extension area */
    if (checked_consume32(input_buffer,
                          &cursor,
                          input_len,
                          &extension_area)) {
        return -1;
    }
    cursor += extension_area;
    if (extension_area) {
        PyErr_SetString(PyExc_ValueError, "non-zero extension area length");
        return -1;
    }

    if (allocate_outarrays(ncolumns, column_types, outarrays, outmasks)) {
        return -1;
    }


    while (true) {
        int16_t field_count;

        if (checked_consume16(input_buffer,
                              &cursor,
                              input_len,
                              (uint16_t*) &field_count)) {
            free_outarrays(ncolumns,
                           row_count,
                           column_types,
                           outarrays,
                           outmasks);
            return -1;
        }

        if (field_count == -1) {
            /* field_count == -1 signals the end of the input data */
            break;
        }

        if (field_count != ncolumns) {
            PyErr_Format(PyExc_ValueError,
                         "mismatched field_count and ncolumns on row %zu:"
                         " %d != %d",
                         row_count,
                         field_count,
                         ncolumns);
            return -1;
        }

        if (have_oids(flags)) {
            uint32_t oid;
            if (checked_consume32(input_buffer,
                                  &cursor,
                                  input_len,
                                  &oid)) {
                free_outarrays(ncolumns,
                               row_count,
                               column_types,
                               outarrays,
                               outmasks);
                return -1;
            }
        }

        /* advance the row count; grow arrays if needed */
        if (row_count++ == allocated_rows) {
            if (grow_outarrays(ncolumns,
                               &allocated_rows,
                               column_types,
                               outarrays,
                               outmasks)) {
                return -1;
            }
        }

        for (uint_fast16_t n = 0; n < ncolumns; ++n) {
            const warp_prism_type* column_type = column_types[n];
            int32_t datalen;
            size_t row_ix = row_count - 1;
            char* column_buffer = &outarrays[n][row_ix * column_type->size];

            if (checked_consume32(input_buffer,
                                  &cursor,
                                  input_len,
                                  (uint32_t*) &datalen)) {
                goto error;
            }

            if (!(outmasks[n][row_ix] = (datalen != -1))) {
                if (column_type->write_null(column_buffer, column_type->size)) {
                    goto error;
                }

                /* no value bytes follow a null */
                continue;
            }

            if (assert_can_consume(datalen, cursor, input_len) ||
                column_type->parse(column_buffer,
                                   &input_buffer[cursor],
                                   datalen)) {
                goto error;
            }
            cursor += datalen;
            continue;

        error:
            /* Write a NULL of the correct size to all of the columns that
               have not yet been written. This ensures that we can properly
               cleanup all of the column arrays with `free_outarrays`. */
            for (; n < ncolumns; ++n) {
                const warp_prism_type* type = column_types[n];
                char* buffer = &outarrays[n][row_ix * column_type->size];
                memset(buffer, 0, type->size);
            }
            free_outarrays(ncolumns,
                           row_count,
                           column_types,
                           outarrays,
                           outmasks);
            return -1;
        }
    }
    *written_rows = row_count;
    return 0;
}

typedef struct {
    char* buffer;
    const warp_prism_type* type;
    size_t rowcount;
} capsule_contents;

static void free_acapsule(PyObject* capsule) {
    capsule_contents* c = PyCapsule_GetPointer(capsule, NULL);

    if (c) {
        c->type->free(c->buffer, c->rowcount);
        PyMem_Free(c);
    }
}

static void free_mcapsule(PyObject* capsule) {
    PyMem_Free(PyCapsule_GetPointer(capsule, NULL));
}

static PyObject* warp_prism_to_arrays(PyObject* self __attribute__((unused)),
                                      PyObject* args) {
    Py_buffer view;
    PyObject* pytypeids;
    Py_ssize_t ncolumns;
    const warp_prism_type** types = NULL;
    char** outarrays = NULL;
    bool** outmasks = NULL;;
    Py_ssize_t n;
    size_t written_rows;
    PyObject* out;

    if (PyTuple_GET_SIZE(args) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "expected exactly 2 arguments (buffer, type_ids)");
        return NULL;
    }

    pytypeids = PyTuple_GET_ITEM(args, 1);

    if (!PyTuple_Check(pytypeids)) {
        PyErr_SetString(PyExc_TypeError, "type_ids must be a tuple");
        return NULL;
    }
    ncolumns = PyTuple_GET_SIZE(pytypeids);
    if (ncolumns > UINT16_MAX) {
        PyErr_SetString(PyExc_ValueError, "column count must fit in uint16_t");
        return NULL;
    }

    if (!(outarrays = PyMem_Malloc(sizeof(char*) * ncolumns))) {
        goto free_arrays;
    }
    if (!(outmasks = PyMem_Malloc(sizeof(bool*) * ncolumns))) {
        goto free_arrays;
    }
    if (!(types = PyMem_Malloc(sizeof(warp_prism_type*) * ncolumns))) {
        goto free_arrays;
    }

    for (n = 0; n < ncolumns; ++n) {
        unsigned long id_ix;

        id_ix = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(pytypeids, n));
        if (PyErr_Occurred() || id_ix > max_typeid) {
            goto free_arrays;
        }

        types[n] = typeids[id_ix];

    }

    if (!(out = PyTuple_New(ncolumns))) {
        goto free_arrays;
    }

    if (PyObject_GetBuffer(PyTuple_GET_ITEM(args, 0),
                           &view,
                           PyBUF_CONTIG_RO)) {
        return NULL;
    }

    if (warp_prism_read_binary_results(view.buf,
                                       view.len,
                                       ncolumns,
                                       types,
                                       &written_rows,
                                       outarrays,
                                       outmasks)) {
        PyBuffer_Release(&view);
        goto free_arrays;
    }
    PyBuffer_Release(&view);

    for (n = 0;n < ncolumns; ++n) {
        capsule_contents* ac;
        PyObject* acapsule;
        PyObject* mcapsule;
        PyObject* andaray;
        PyObject* mndarray;
        PyObject* pair;

        Py_INCREF(types[n]->dtype);
        if (!(andaray = PyArray_NewFromDescr(&PyArray_Type,
                                             types[n]->dtype,
                                             1,
                                             (npy_intp*) &written_rows,
                                             NULL,
                                             outarrays[n],
                                             NPY_ARRAY_CARRAY,
                                             NULL))) {
            Py_DECREF(out);
            goto clear_arrays;
        }

        if (!(ac = PyMem_Malloc(sizeof(capsule_contents)))) {
            Py_DECREF(andaray);
            Py_DECREF(out);
            goto clear_arrays;
        }

        ac->buffer = outarrays[n];
        ac->type = types[n];
        ac->rowcount = written_rows;

        if (!(acapsule = PyCapsule_New(ac, NULL, free_acapsule))) {
            PyMem_Free(ac);
            Py_DECREF(andaray);
            Py_DECREF(out);
            goto clear_arrays;
        }

        if (PyArray_SetBaseObject((PyArrayObject*) andaray, acapsule)) {
            Py_DECREF(acapsule);
            Py_DECREF(andaray);
            Py_DECREF(out);
            goto clear_arrays;
        }

        if (!(mndarray = PyArray_SimpleNewFromData(1,
                                                   (npy_intp*) &written_rows,
                                                   NPY_BOOL,
                                                   outmasks[n]))) {
            Py_DECREF(andaray);
            Py_DECREF(out);
            goto clear_arrays;
        }

        if (!(mcapsule = PyCapsule_New(outmasks[n], NULL, free_mcapsule))) {
            Py_DECREF(andaray);
            Py_DECREF(mndarray);
            Py_DECREF(out);
            goto clear_arrays;
        }

        if (PyArray_SetBaseObject((PyArrayObject*) mndarray, mcapsule)) {
            Py_DECREF(andaray);
            Py_DECREF(mndarray);
            Py_DECREF(mcapsule);
            Py_DECREF(out);
            goto clear_arrays;
        }

        if (!(pair = PyTuple_New(2))) {
            Py_DECREF(andaray);
            Py_DECREF(mndarray);
            Py_DECREF(out);
            goto clear_arrays;
        }

        PyTuple_SET_ITEM(pair, 0, andaray);
        PyTuple_SET_ITEM(pair, 1, mndarray);
        PyTuple_SET_ITEM(out, n, pair);
    }

    return out;

clear_arrays:
    free_outarrays(ncolumns, written_rows, types, outarrays, outmasks);
free_arrays:
    PyMem_Free(outarrays);
    PyMem_Free(outmasks);
    PyMem_Free(types);
    return NULL;
}

PyMethodDef methods[] = {
    {"raw_to_arrays", (PyCFunction) warp_prism_to_arrays, METH_VARARGS, NULL},
    {NULL},
};

static struct PyModuleDef _warp_prism_module = {
    PyModuleDef_HEAD_INIT,
    "warp_prism._warp_prism",
    "",
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__warp_prism(void) {
    PyObject* m;
    PyObject* typeid_map;
    PyObject* signature_ob;

    /* This is needed to setup the numpy C-API. */
    import_array();

    if (!(typeid_map = PyDict_New())) {
        return NULL;
    }

    for (size_t n = 0; n < max_typeid; ++n) {
        PyObject* dtype_name_ob;
        PyObject* n_ob;
        int err;


        if (!(dtype_name_ob = PyUnicode_FromString(typeids[n]->dtype_name))) {
            Py_DECREF(typeid_map);
            return NULL;
        }

        if (!PyArray_DescrConverter(dtype_name_ob,
                                    (PyArray_Descr**) &typeids[n]->dtype)) {
            Py_DECREF(dtype_name_ob);
            Py_DECREF(typeid_map);
            return NULL;
        }


        if (!(n_ob = PyLong_FromLong(n))) {
            Py_DECREF(dtype_name_ob);
            Py_DECREF(typeid_map);
            return NULL;
        }

        err = PyDict_SetItem(typeid_map, dtype_name_ob, n_ob);
        Py_DECREF(dtype_name_ob);
        Py_DECREF(n_ob);
        if (err) {
            Py_DECREF(typeid_map);
            return NULL;
        }
    }

    if (!(m = PyModule_Create(&_warp_prism_module))) {
        Py_DECREF(typeid_map);
        return NULL;
    }

    if (PyModule_AddObject(m, "typeid_map", typeid_map)) {
        Py_DECREF(typeid_map);
        Py_DECREF(m);
        return NULL;
    }

    if (!(signature_ob = PyBytes_FromStringAndSize(signature, signature_len))) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObject(m, "postgres_signature", signature_ob)) {
        Py_DECREF(signature_ob);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
