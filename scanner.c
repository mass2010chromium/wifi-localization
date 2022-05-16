#include <iwlib.h>
#include <stdlib.h>
#include <net/ethernet.h> // struct ether_addr
#include "structures/Vector.h"

/*------------------------------------------------------------------*/
/*
 * Output the link statistics, taking care of formating
 */
void
custom_print_stats(char *    buffer,
        int    buflen,
        const iwqual *  qual,
        const iwrange *  range,
        int    has_range)
{
    int    len;

    /* People are very often confused by the 8 bit arithmetic happening
     * here.
     * All the values here are encoded in a 8 bit integer. 8 bit integers
     * are either unsigned [0 ; 255], signed [-128 ; +127] or
     * negative [-255 ; 0].
     * Further, on 8 bits, 0x100 == 256 == 0.
     *
     * Relative/percent values are always encoded unsigned, between 0 and 255.
     * Absolute/dBm values are always encoded between -192 and 63.
     * (Note that up to version 28 of Wireless Tools, dBm used to be
     *  encoded always negative, between -256 and -1).
     *
     * How do we separate relative from absolute values ?
     * The old way is to use the range to do that. As of WE-19, we have
     * an explicit IW_QUAL_DBM flag in updated...
     * The range allow to specify the real min/max of the value. As the
     * range struct only specify one bound of the value, we assume that
     * the other bound is 0 (zero).
     * For relative values, range is [0 ; range->max].
     * For absolute values, range is [range->max ; 63].
     *
     * Let's take two example :
     * 1) value is 75%. qual->value = 75 ; range->max_qual.value = 100
     * 2) value is -54dBm. noise floor of the radio is -104dBm.
     *    qual->value = -54 = 202 ; range->max_qual.value = -104 = 152
     *
     * Jean II
     */

    /* Just do it...
     * The old way to detect dBm require both the range and a non-null
     * level (which confuse the test). The new way can deal with level of 0
     * because it does an explicit test on the flag. */
    if(has_range && ((qual->level != 0)
                || (qual->updated & (IW_QUAL_DBM | IW_QUAL_RCPI))))
    {
        /* Deal with quality : always a relative value */
        if(!(qual->updated & IW_QUAL_QUAL_INVALID))
        {
            len = snprintf(buffer, buflen, "Quality%c%d/%d  ",
                    qual->updated & IW_QUAL_QUAL_UPDATED ? '=' : ':',
                    qual->qual, range->max_qual.qual);
            buffer += len;
            buflen -= len;
        }

        /* Check if the statistics are in RCPI (IEEE 802.11k) */
        if(qual->updated & IW_QUAL_RCPI)
        {
            /* Deal with signal level in RCPI */
            /* RCPI = int{(Power in dBm +110)*2} for 0dbm > Power > -110dBm */
            if(!(qual->updated & IW_QUAL_LEVEL_INVALID))
            {
                double  rcpilevel = (qual->level / 2.0) - 110.0;
                len = snprintf(buffer, buflen, "Signal level A %c%g dBm  ",
                        qual->updated & IW_QUAL_LEVEL_UPDATED ? '=' : ':',
                        rcpilevel);
                buffer += len;
                buflen -= len;
            }

            /* Deal with noise level in dBm (absolute power measurement) */
            if(!(qual->updated & IW_QUAL_NOISE_INVALID))
            {
                double  rcpinoise = (qual->noise / 2.0) - 110.0;
                len = snprintf(buffer, buflen, "Noise level%c%g dBm",
                        qual->updated & IW_QUAL_NOISE_UPDATED ? '=' : ':',
                        rcpinoise);
            }
        }
        else if((qual->updated & IW_QUAL_DBM)
                || (qual->level > range->max_qual.level))
        {
            /* Deal with signal level in dBm  (absolute power measurement) */
            if(!(qual->updated & IW_QUAL_LEVEL_INVALID))
            {
                int  dblevel = qual->level;
                /* Implement a range for dBm [-192; 63] */
                if(qual->level >= 64)
                    dblevel -= 0x100;
                len = snprintf(buffer, buflen, "Signal level B %c%d dBm  ",
                        qual->updated & IW_QUAL_LEVEL_UPDATED ? '=' : ':',
                        dblevel);
                buffer += len;
                buflen -= len;
            }

            /* Deal with noise level in dBm (absolute power measurement) */
            if(!(qual->updated & IW_QUAL_NOISE_INVALID))
            {
                int  dbnoise = qual->noise;
                /* Implement a range for dBm [-192; 63] */
                if(qual->noise >= 64)
                    dbnoise -= 0x100;
                len = snprintf(buffer, buflen, "Noise level%c%d dBm",
                        qual->updated & IW_QUAL_NOISE_UPDATED ? '=' : ':',
                        dbnoise);
            }
        }
        else
        {
            /* Deal with signal level as relative value (0 -> max) */
            if(!(qual->updated & IW_QUAL_LEVEL_INVALID))
            {
                len = snprintf(buffer, buflen, "Signal level%c%d/%d  ",
                        qual->updated & IW_QUAL_LEVEL_UPDATED ? '=' : ':',
                        qual->level, range->max_qual.level);
                buffer += len;
                buflen -= len;
            }

            /* Deal with noise level as relative value (0 -> max) */
            if(!(qual->updated & IW_QUAL_NOISE_INVALID))
            {
                len = snprintf(buffer, buflen, "Noise level%c%d/%d",
                        qual->updated & IW_QUAL_NOISE_UPDATED ? '=' : ':',
                        qual->noise, range->max_qual.noise);
            }
        }
    }
    else
    {
        /* We can't read the range, so we don't know... */
        snprintf(buffer, buflen,
                "Quality:%d  Signal level:%d  Noise level:%d",
                qual->qual, qual->level, qual->noise);
    }
}

/****************************** TYPES ******************************/

/*
 * Scan state and meta-information, used to decode events...
 */
typedef struct iwscan_state
{
    /* State */
    int      ap_num;    /* Access Point number 1->N */
    int      val_index;  /* Value in table 0->(N-1) */
} iwscan_state;

/*
 * Bit to name mapping
 */
typedef struct iwmask_name
{
    unsigned int  mask;  /* bit mask for the value */
    const char *  name;  /* human readable name for the value */
} iwmask_name;

/*
 * Types of authentication parameters
 */
typedef struct iw_auth_descr
{
    int        value;    /* Type of auth value */
    const char *      label;    /* User readable version */
    const struct iwmask_name *  names;    /* Names for this value */
    const int      num_names;  /* Number of names */
} iw_auth_descr;

/**************************** CONSTANTS ****************************/

#define IW_SCAN_HACK    0x8000

/* ------------------------ WPA CAPA NAMES ------------------------ */
/*
 * This is the user readable name of a bunch of WPA constants in wireless.h
 * Maybe this should go in iwlib.c ?
 */

#ifndef WE_ESSENTIAL
#define IW_ARRAY_LEN(x) (sizeof(x)/sizeof((x)[0]))

//static const struct iwmask_name iw_enc_mode_name[] = {
//  { IW_ENCODE_RESTRICTED,  "restricted" },
//  { IW_ENCODE_OPEN,    "open" },
//};
//#define  IW_ENC_MODE_NUM    IW_ARRAY_LEN(iw_enc_mode_name)

static const struct iwmask_name iw_auth_capa_name[] = {
    { IW_ENC_CAPA_WPA,    "WPA" },
    { IW_ENC_CAPA_WPA2,    "WPA2" },
    { IW_ENC_CAPA_CIPHER_TKIP,  "CIPHER-TKIP" },
    { IW_ENC_CAPA_CIPHER_CCMP,  "CIPHER-CCMP" },
};
#define  IW_AUTH_CAPA_NUM  IW_ARRAY_LEN(iw_auth_capa_name)

static const struct iwmask_name iw_auth_cypher_name[] = {
    { IW_AUTH_CIPHER_NONE,  "none" },
    { IW_AUTH_CIPHER_WEP40,  "WEP-40" },
    { IW_AUTH_CIPHER_TKIP,  "TKIP" },
    { IW_AUTH_CIPHER_CCMP,  "CCMP" },
    { IW_AUTH_CIPHER_WEP104,  "WEP-104" },
};
#define  IW_AUTH_CYPHER_NUM  IW_ARRAY_LEN(iw_auth_cypher_name)

static const struct iwmask_name iw_wpa_ver_name[] = {
    { IW_AUTH_WPA_VERSION_DISABLED,  "disabled" },
    { IW_AUTH_WPA_VERSION_WPA,    "WPA" },
    { IW_AUTH_WPA_VERSION_WPA2,    "WPA2" },
};
#define  IW_WPA_VER_NUM    IW_ARRAY_LEN(iw_wpa_ver_name)

static const struct iwmask_name iw_auth_key_mgmt_name[] = {
    { IW_AUTH_KEY_MGMT_802_1X,  "802.1x" },
    { IW_AUTH_KEY_MGMT_PSK,  "PSK" },
};
#define  IW_AUTH_KEY_MGMT_NUM  IW_ARRAY_LEN(iw_auth_key_mgmt_name)

static const struct iwmask_name iw_auth_alg_name[] = {
    { IW_AUTH_ALG_OPEN_SYSTEM,  "open" },
    { IW_AUTH_ALG_SHARED_KEY,  "shared-key" },
    { IW_AUTH_ALG_LEAP,    "LEAP" },
};
#define  IW_AUTH_ALG_NUM    IW_ARRAY_LEN(iw_auth_alg_name)

static const struct iw_auth_descr  iw_auth_settings[] = {
    { IW_AUTH_WPA_VERSION, "WPA version", iw_wpa_ver_name, IW_WPA_VER_NUM },
    { IW_AUTH_KEY_MGMT, "Key management", iw_auth_key_mgmt_name, IW_AUTH_KEY_MGMT_NUM },
    { IW_AUTH_CIPHER_PAIRWISE, "Pairwise cipher", iw_auth_cypher_name, IW_AUTH_CYPHER_NUM },
    { IW_AUTH_CIPHER_GROUP, "Pairwise cipher", iw_auth_cypher_name, IW_AUTH_CYPHER_NUM },
    { IW_AUTH_TKIP_COUNTERMEASURES, "TKIP countermeasures", NULL, 0 },
    { IW_AUTH_DROP_UNENCRYPTED, "Drop unencrypted", NULL, 0 },
    { IW_AUTH_80211_AUTH_ALG, "Authentication algorithm", iw_auth_alg_name, IW_AUTH_ALG_NUM },
    { IW_AUTH_RX_UNENCRYPTED_EAPOL, "Receive unencrypted EAPOL", NULL, 0 },
    { IW_AUTH_ROAMING_CONTROL, "Roaming control", NULL, 0 },
    { IW_AUTH_PRIVACY_INVOKED, "Privacy invoked", NULL, 0 },
};
#define  IW_AUTH_SETTINGS_NUM    IW_ARRAY_LEN(iw_auth_settings)

/* Values for the IW_ENCODE_ALG_* returned by SIOCSIWENCODEEXT */
static const char *  iw_encode_alg_name[] = {
    "none",
    "WEP",
    "TKIP",
    "CCMP",
    "unknown"
};
#define  IW_ENCODE_ALG_NUM    IW_ARRAY_LEN(iw_encode_alg_name)

#ifndef IW_IE_CIPHER_NONE
/* Cypher values in GENIE (pairwise and group) */
#define IW_IE_CIPHER_NONE  0
#define IW_IE_CIPHER_WEP40  1
#define IW_IE_CIPHER_TKIP  2
#define IW_IE_CIPHER_WRAP  3
#define IW_IE_CIPHER_CCMP  4
#define IW_IE_CIPHER_WEP104  5
/* Key management in GENIE */
#define IW_IE_KEY_MGMT_NONE  0
#define IW_IE_KEY_MGMT_802_1X  1
#define IW_IE_KEY_MGMT_PSK  2
#endif  /* IW_IE_CIPHER_NONE */

/* Values for the IW_IE_CIPHER_* in GENIE */
static const char *  iw_ie_cypher_name[] = {
    "none",
    "WEP-40",
    "TKIP",
    "WRAP",
    "CCMP",
    "WEP-104",
};
#define  IW_IE_CYPHER_NUM  IW_ARRAY_LEN(iw_ie_cypher_name)

/* Values for the IW_IE_KEY_MGMT_* in GENIE */
static const char *  iw_ie_key_mgmt_name[] = {
    "none",
    "802.1x",
    "PSK",
};
#define  IW_IE_KEY_MGMT_NUM  IW_ARRAY_LEN(iw_ie_key_mgmt_name)

#endif  /* WE_ESSENTIAL */

/************************* WPA SUBROUTINES *************************/

#ifndef WE_ESSENTIAL
/*------------------------------------------------------------------*/
/*
 * Print all names corresponding to a mask.
 * This may want to be used in iw_print_retry_value() ?
 */
    static void 
iw_print_mask_name(unsigned int      mask,
        const struct iwmask_name  names[],
        const unsigned int    num_names,
        const char *      sep)
{
    unsigned int  i;

    /* Print out all names for the bitmask */
    for(i = 0; i < num_names; i++)
    {
        if(mask & names[i].mask)
        {
            /* Print out */
            printf("%s%s", sep, names[i].name);
            /* Remove the bit from the mask */
            mask &= ~names[i].mask;
        }
    }
    /* If there is unconsumed bits... */
    if(mask != 0)
        printf("%sUnknown", sep);
}

/*------------------------------------------------------------------*/
/*
 * Print the name corresponding to a value, with overflow check.
 */
    static void
iw_print_value_name(unsigned int    value,
        const char *    names[],
        const unsigned int    num_names)
{
    if(value >= num_names)
        printf(" unknown (%d)", value);
    else
        printf(" %s", names[value]);
}

/*------------------------------------------------------------------*/
/*
 * Parse, and display the results of an unknown IE.
 *
 */
    static void 
iw_print_ie_unknown(unsigned char *  iebuf,
        int      buflen)
{
    int  ielen = iebuf[1] + 2;
    int  i;

    if(ielen > buflen)
        ielen = buflen;

    printf("Unknown: ");
    for(i = 0; i < ielen; i++)
        printf("%02X", iebuf[i]);
    printf("\n");
}

/*------------------------------------------------------------------*/
/*
 * Parse, and display the results of a WPA or WPA2 IE.
 *
 */
    static inline void 
iw_print_ie_wpa(unsigned char *  iebuf,
        int    buflen)
{
    int      ielen = iebuf[1] + 2;
    int      offset = 2;  /* Skip the IE id, and the length. */
    unsigned char    wpa1_oui[3] = {0x00, 0x50, 0xf2};
    unsigned char    wpa2_oui[3] = {0x00, 0x0f, 0xac};
    unsigned char *  wpa_oui;
    int      i;
    uint16_t    ver = 0;
    uint16_t    cnt = 0;

    if(ielen > buflen)
        ielen = buflen;

#ifdef DEBUG
    /* Debugging code. In theory useless, because it's debugged ;-) */
    printf("IE raw value %d [%02X", buflen, iebuf[0]);
    for(i = 1; i < buflen; i++)
        printf(":%02X", iebuf[i]);
    printf("]\n");
#endif

    switch(iebuf[0])
    {
        case 0x30:    /* WPA2 */
            /* Check if we have enough data */
            if(ielen < 4)
            {
                iw_print_ie_unknown(iebuf, buflen);
                return;
            }

            wpa_oui = wpa2_oui;
            break;

        case 0xdd:    /* WPA or else */
            wpa_oui = wpa1_oui;

            /* Not all IEs that start with 0xdd are WPA. 
             * So check that the OUI is valid. Note : offset==2 */
            if((ielen < 8)
                    || (memcmp(&iebuf[offset], wpa_oui, 3) != 0)
                    || (iebuf[offset + 3] != 0x01))
            {
                iw_print_ie_unknown(iebuf, buflen);
                return;
            }

            /* Skip the OUI type */
            offset += 4;
            break;

        default:
            return;
    }

    /* Pick version number (little endian) */
    ver = iebuf[offset] | (iebuf[offset + 1] << 8);
    offset += 2;

    if(iebuf[0] == 0xdd)
        printf("WPA Version %d\n", ver);
    if(iebuf[0] == 0x30)
        printf("IEEE 802.11i/WPA2 Version %d\n", ver);

    /* From here, everything is technically optional. */

    /* Check if we are done */
    if(ielen < (offset + 4))
    {
        /* We have a short IE.  So we should assume TKIP/TKIP. */
        printf("                        Group Cipher : TKIP\n");
        printf("                        Pairwise Cipher : TKIP\n");
        return;
    }

    /* Next we have our group cipher. */
    if(memcmp(&iebuf[offset], wpa_oui, 3) != 0)
    {
        printf("                        Group Cipher : Proprietary\n");
    }
    else
    {
        printf("                        Group Cipher :");
        iw_print_value_name(iebuf[offset+3],
                iw_ie_cypher_name, IW_IE_CYPHER_NUM);
        printf("\n");
    }
    offset += 4;

    /* Check if we are done */
    if(ielen < (offset + 2))
    {
        /* We don't have a pairwise cipher, or auth method. Assume TKIP. */
        printf("                        Pairwise Ciphers : TKIP\n");
        return;
    }

    /* Otherwise, we have some number of pairwise ciphers. */
    cnt = iebuf[offset] | (iebuf[offset + 1] << 8);
    offset += 2;
    printf("                        Pairwise Ciphers (%d) :", cnt);

    if(ielen < (offset + 4*cnt))
        return;

    for(i = 0; i < cnt; i++)
    {
        if(memcmp(&iebuf[offset], wpa_oui, 3) != 0)
        {
            printf(" Proprietary");
        }
        else
        {
            iw_print_value_name(iebuf[offset+3],
                    iw_ie_cypher_name, IW_IE_CYPHER_NUM);
        }
        offset+=4;
    }
    printf("\n");

    /* Check if we are done */
    if(ielen < (offset + 2))
        return;

    /* Now, we have authentication suites. */
    cnt = iebuf[offset] | (iebuf[offset + 1] << 8);
    offset += 2;
    printf("                        Authentication Suites (%d) :", cnt);

    if(ielen < (offset + 4*cnt))
        return;

    for(i = 0; i < cnt; i++)
    {
        if(memcmp(&iebuf[offset], wpa_oui, 3) != 0)
        {
            printf(" Proprietary");
        }
        else
        {
            iw_print_value_name(iebuf[offset+3],
                    iw_ie_key_mgmt_name, IW_IE_KEY_MGMT_NUM);
        }
        offset+=4;
    }
    printf("\n");

    /* Check if we are done */
    if(ielen < (offset + 1))
        return;

    /* Otherwise, we have capabilities bytes.
     * For now, we only care about preauth which is in bit position 1 of the
     * first byte.  (But, preauth with WPA version 1 isn't supposed to be 
     * allowed.) 8-) */
    if(iebuf[offset] & 0x01)
    {
        printf("                       Preauthentication Supported\n");
    }
}

/*------------------------------------------------------------------*/
/*
 * Process a generic IE and display the info in human readable form
 * for some of the most interesting ones.
 * For now, we only decode the WPA IEs.
 */
    static inline void
iw_print_gen_ie(unsigned char *  buffer,
        int    buflen)
{
    int offset = 0;

    /* Loop on each IE, each IE is minimum 2 bytes */
    while(offset <= (buflen - 2))
    {
        printf("                    IE: ");

        /* Check IE type */
        switch(buffer[offset])
        {
            case 0xdd:  /* WPA1 (and other) */
            case 0x30:  /* WPA2 */
                iw_print_ie_wpa(buffer + offset, buflen);
                break;
            default:
                iw_print_ie_unknown(buffer + offset, buflen);
        }
        /* Skip over this IE to the next one in the list. */
        offset += buffer[offset+1] + 2;
    }
}
#endif  /* WE_ESSENTIAL */

/***************************** SCANNING *****************************/
/*
 * This one behave quite differently from the others
 *
 * Note that we don't use the scanning capability of iwlib (functions
 * iw_process_scan() and iw_scan()). The main reason is that
 * iw_process_scan() return only a subset of the scan data to the caller,
 * for example custom elements and bitrates are ommited. Here, we
 * do the complete job...
 */

struct scan_result {
    int id;                 // internal ID (scan return order)
    char* protocol_name;    // protocol name (unsure). possibly null
    int32_t nwid;           // network id (or domain - the cell)
    struct ether_addr address;  // access point address
    double freq;            /// frequency of operation
    int channel;            // channel corresponding to frequency (computed)
    uint32_t mode;          // operation mode
    char* essid;            // ESSID of this access point; possibly null
    Vector bitrates;        // list of all bitrates i guess...? dunno why there are sometimes 2 lines of bitrates
    int signal_strength;    // dBm; 0 for invalid
    int noise;              // dBm; 0 for invalid
    int quality_val;        // quality value (relative to quality_max)
    int quality_max;        // quality maximum (denominator)
};
typedef struct scan_result scan_result;

scan_result* make_scan_result() {
    scan_result* res = malloc(sizeof(scan_result));
    memset(res, 0, sizeof(res));
    inplace_make_Vector(&res->bitrates, 10);
    return res;
}

/*------------------------------------------------------------------*/
/*
 * Parse the link statistics.
 */
void custom_parse_stats(const iwqual* qual, const iwrange* range, int has_range, scan_result* result) {
    /* People are very often confused by the 8 bit arithmetic happening
     * here.
     * All the values here are encoded in a 8 bit integer. 8 bit integers
     * are either unsigned [0 ; 255], signed [-128 ; +127] or
     * negative [-255 ; 0].
     * Further, on 8 bits, 0x100 == 256 == 0.
     *
     * Relative/percent values are always encoded unsigned, between 0 and 255.
     * Absolute/dBm values are always encoded between -192 and 63.
     * (Note that up to version 28 of Wireless Tools, dBm used to be
     *  encoded always negative, between -256 and -1).
     *
     * How do we separate relative from absolute values ?
     * The old way is to use the range to do that. As of WE-19, we have
     * an explicit IW_QUAL_DBM flag in updated...
     * The range allow to specify the real min/max of the value. As the
     * range struct only specify one bound of the value, we assume that
     * the other bound is 0 (zero).
     * For relative values, range is [0 ; range->max].
     * For absolute values, range is [range->max ; 63].
     *
     * Let's take two example :
     * 1) value is 75%. qual->value = 75 ; range->max_qual.value = 100
     * 2) value is -54dBm. noise floor of the radio is -104dBm.
     *    qual->value = -54 = 202 ; range->max_qual.value = -104 = 152
     *
     * Jean II
     */

    /* Just do it...
     * The old way to detect dBm require both the range and a non-null
     * level (which confuse the test). The new way can deal with level of 0
     * because it does an explicit test on the flag. */
    if (has_range && ((qual->level != 0) || (qual->updated & (IW_QUAL_DBM | IW_QUAL_RCPI)))) {
        /* Deal with quality : always a relative value */
        if (!(qual->updated & IW_QUAL_QUAL_INVALID)) {
            result->quality_val = qual->qual;
            result->quality_max = range->max_qual.qual;
        }

        /* Check if the statistics are in RCPI (IEEE 802.11k) */
        if (qual->updated & IW_QUAL_RCPI) {
            /* Deal with signal level in RCPI */
            /* RCPI = int{(Power in dBm +110)*2} for 0dbm > Power > -110dBm */
            if (!(qual->updated & IW_QUAL_LEVEL_INVALID)) {
                double  rcpilevel = (qual->level / 2.0) - 110.0;
                result->signal_strength = (int)rcpilevel;
            }

            /* Deal with noise level in dBm (absolute power measurement) */
            if (!(qual->updated & IW_QUAL_NOISE_INVALID)) {
                double  rcpinoise = (qual->noise / 2.0) - 110.0;
                result->noise = (int)rcpinoise;
            }
        }
        else if ((qual->updated & IW_QUAL_DBM) || (qual->level > range->max_qual.level)) {
            /* Deal with signal level in dBm  (absolute power measurement) */
            if (!(qual->updated & IW_QUAL_LEVEL_INVALID)) {
                int  dblevel = qual->level;
                /* Implement a range for dBm [-192; 63] */
                if (qual->level >= 64) { dblevel -= 0x100; }
                result->signal_strength = dblevel;
            }

            /* Deal with noise level in dBm (absolute power measurement) */
            if (!(qual->updated & IW_QUAL_NOISE_INVALID)) {
                int  dbnoise = qual->noise;
                /* Implement a range for dBm [-192; 63] */
                if (qual->noise >= 64) { dbnoise -= 0x100; }
                result->noise = dbnoise;
            }
        }
        // Ignoring relative measures.
    }
    // If ranges not found, ignore entirely.
}


/*------------------------------------------------------------------*/
/*
 * Add scanned elements to vector.
 */
static inline void
parse_scanning_token(struct stream_descr *  stream,  /* Stream of events */
        struct iw_event *    event,  /* Extracted token */
        struct iwscan_state *  state,
        struct iw_range *  iw_range,  /* Range info */
        int    has_range,
        Vector* results)
{
    char    buffer[128];  /* Temporary buffer */

    if (event->cmd == SIOCGIWAP) {
        // Handle this one separately since it requires making a new struct.
        scan_result* res = make_scan_result();
        memcpy(&res->address, &event->u.ap_addr, sizeof(struct ether_addr));
        res->id = state->ap_num;
        state->ap_num++;
        Vector_push(results, res);
    }
    scan_result* current = results->elements[results->size-1];

    /* Now, let's decode the event */
    switch(event->cmd)
    {
        case SIOCGIWAP:
            // access point address (handled above)
            break;
        case SIOCGIWNWID:
            // network id
            if (!event->u.nwid.disabled) {
                current->nwid = event->u.nwid.value;
            }
            break;
        case SIOCGIWFREQ:
            // frequency and channel
            {
                double    freq;      /* Frequency/channel */
                int    channel = -1;    /* Converted to channel */
                freq = iw_freq2float(&(event->u.freq));
                /* Convert to channel if possible */
                if (has_range) {
                    channel = iw_freq_to_channel(freq, iw_range);
                }
                current->freq = freq;
                current->channel = channel;
            }
            break;
        case SIOCGIWMODE:
            // operation mode
            /* Note : event->u.mode is unsigned, no need to check <= 0 */
            if(event->u.mode >= IW_NUM_OPER_MODE)
                event->u.mode = IW_NUM_OPER_MODE;
            current->mode = event->u.mode;
            break;
        case SIOCGIWNAME:
            // protocol name
            current->protocol_name = strdup(event->u.name);
            break;
        case SIOCGIWESSID:
            // essid
            {
                char* essid = malloc(IW_ESSID_MAX_SIZE+1);
                memset(essid, '\0', sizeof(essid));
                if ((event->u.essid.pointer) && (event->u.essid.length)) {
                    memcpy(essid, event->u.essid.pointer, event->u.essid.length);
                }
                if (event->u.essid.flags) {
                    // ESSID index information ignored.
                    current->essid = essid;
                } else {
                    free(essid);
                }
            }
            break;
        case SIOCGIWENCODE:
            // wdym encryption keys? not handling!
            break;
        case SIOCGIWRATE:
            // bit rate
            Vector_push(&current->bitrates, event->u.bitrate.value);
            break;
        case SIOCGIWMODUL:
            // modulations: ignored
            break;
        case IWEVQUAL:
            custom_parse_stats(&event->u.qual, iw_range, has_range, current);
            break;
#ifndef WE_ESSENTIAL
        case IWEVGENIE:
            /* Informations Elements are complex, let's do only some of them */
            // actually we'll just skip these
            break;
#endif  /* WE_ESSENTIAL */
        case IWEVCUSTOM:
            // custom parameter (ignored)
            break;
        default:
            // unrecognized
            break;
    }  /* switch(event->cmd) */
}

/*------------------------------------------------------------------*/
/*
 * Print one element from the scanning results
 */
static inline void
print_scanning_token(struct stream_descr *  stream,  /* Stream of events */
        struct iw_event *    event,  /* Extracted token */
        struct iwscan_state *  state,
        struct iw_range *  iw_range,  /* Range info */
        int    has_range)
{
    char    buffer[128];  /* Temporary buffer */

    /* Now, let's decode the event */
    switch(event->cmd)
    {
        case SIOCGIWAP:
            printf("          Cell %02d - Address: %s\n", state->ap_num,
                    iw_saether_ntop(&event->u.ap_addr, buffer));
            state->ap_num++;
            break;
        case SIOCGIWNWID:
            if(event->u.nwid.disabled)
                printf("                    NWID:off/any\n");
            else
                printf("                    NWID:%X\n", event->u.nwid.value);
            break;
        case SIOCGIWFREQ:
            {
                double    freq;      /* Frequency/channel */
                int    channel = -1;    /* Converted to channel */
                freq = iw_freq2float(&(event->u.freq));
                /* Convert to channel if possible */
                if(has_range)
                    channel = iw_freq_to_channel(freq, iw_range);
                iw_print_freq(buffer, sizeof(buffer),
                        freq, channel, event->u.freq.flags);
                printf("                    %s\n", buffer);
            }
            break;
        case SIOCGIWMODE:
            /* Note : event->u.mode is unsigned, no need to check <= 0 */
            if(event->u.mode >= IW_NUM_OPER_MODE)
                event->u.mode = IW_NUM_OPER_MODE;
            printf("                    Mode:%s\n",
                    iw_operation_mode[event->u.mode]);
            break;
        case SIOCGIWNAME:
            printf("                    Protocol:%-1.16s\n", event->u.name);
            break;
        case SIOCGIWESSID:
            {
                char essid[IW_ESSID_MAX_SIZE+1];
                memset(essid, '\0', sizeof(essid));
                if((event->u.essid.pointer) && (event->u.essid.length))
                    memcpy(essid, event->u.essid.pointer, event->u.essid.length);
                if(event->u.essid.flags)
                {
                    /* Does it have an ESSID index ? */
                    if((event->u.essid.flags & IW_ENCODE_INDEX) > 1)
                        printf("                    ESSID:\"%s\" [%d]\n", essid,
                                (event->u.essid.flags & IW_ENCODE_INDEX));
                    else
                        printf("                    ESSID:\"%s\"\n", essid);
                }
                else
                    printf("                    ESSID:off/any/hidden\n");
            }
            break;
        case SIOCGIWENCODE:
            {
                unsigned char  key[IW_ENCODING_TOKEN_MAX];
                if(event->u.data.pointer)
                    memcpy(key, event->u.data.pointer, event->u.data.length);
                else
                    event->u.data.flags |= IW_ENCODE_NOKEY;
                printf("                    Encryption key:");
                if(event->u.data.flags & IW_ENCODE_DISABLED)
                    printf("off\n");
                else
                {
                    /* Display the key */
                    iw_print_key(buffer, sizeof(buffer), key, event->u.data.length,
                            event->u.data.flags);
                    printf("%s", buffer);

                    /* Other info... */
                    if((event->u.data.flags & IW_ENCODE_INDEX) > 1)
                        printf(" [%d]", event->u.data.flags & IW_ENCODE_INDEX);
                    if(event->u.data.flags & IW_ENCODE_RESTRICTED)
                        printf("   Security mode:restricted");
                    if(event->u.data.flags & IW_ENCODE_OPEN)
                        printf("   Security mode:open");
                    printf("\n");
                }
            }
            break;
        case SIOCGIWRATE:
            if(state->val_index == 0)
                printf("                    Bit Rates:");
            else
                if((state->val_index % 5) == 0)
                    printf("\n                              ");
                else
                    printf("; ");
            iw_print_bitrate(buffer, sizeof(buffer), event->u.bitrate.value);
            printf("%s", buffer);
            /* Check for termination */
            if(stream->value == NULL)
            {
                printf("\n");
                state->val_index = 0;
            }
            else
                state->val_index++;
            break;
        case SIOCGIWMODUL:
            {
                unsigned int  modul = event->u.param.value;
                int    i;
                int    n = 0;
                printf("                    Modulations :");
                for(i = 0; i < IW_SIZE_MODUL_LIST; i++)
                {
                    if((modul & iw_modul_list[i].mask) == iw_modul_list[i].mask)
                    {
                        if((n++ % 8) == 7)
                            printf("\n                        ");
                        else
                            printf(" ; ");
                        printf("%s", iw_modul_list[i].cmd);
                    }
                }
                printf("\n");
            }
            break;
        case IWEVQUAL:
            custom_print_stats(buffer, sizeof(buffer),
                    &event->u.qual, iw_range, has_range);
            printf("                    %s\n", buffer);
            break;
#ifndef WE_ESSENTIAL
        case IWEVGENIE:
            /* Informations Elements are complex, let's do only some of them */
            iw_print_gen_ie(event->u.data.pointer, event->u.data.length);
            break;
#endif  /* WE_ESSENTIAL */
        case IWEVCUSTOM:
            {
                char custom[IW_CUSTOM_MAX+1];
                if((event->u.data.pointer) && (event->u.data.length))
                    memcpy(custom, event->u.data.pointer, event->u.data.length);
                custom[event->u.data.length] = '\0';
                printf("                    Extra:%s\n", custom);
            }
            break;
        default:
            printf("                    (Unknown Wireless Token 0x%04X)\n",
                    event->cmd);
    }  /* switch(event->cmd) */
}

/*------------------------------------------------------------------*/
/*
 * Perform a scanning on one device
 */
    static int
print_scanning_info(int    skfd,
        char *  ifname,
        char *  args[],    /* Command line args */
        int    count)    /* Args count */
{
    struct iwreq    wrq;
    struct iw_scan_req    scanopt;    /* Options for 'set' */
    int      scanflags = 0;    /* Flags for scan */
    unsigned char *  buffer = NULL;    /* Results */
    int      buflen = IW_SCAN_MAX_DATA; /* Min for compat WE<17 */
    struct iw_range  range;
    int      has_range;
    struct timeval  tv;        /* Select timeout */
    int      timeout = 15000000;    /* 15s */

    /* Avoid "Unused parameter" warning */
    args = args; count = count;

    /* Debugging stuff */
    if((IW_EV_LCP_PK2_LEN != IW_EV_LCP_PK_LEN) || (IW_EV_POINT_PK2_LEN != IW_EV_POINT_PK_LEN))
    {
        fprintf(stderr, "*** Please report to jt@hpl.hp.com your platform details\n");
        fprintf(stderr, "*** and the following line :\n");
        fprintf(stderr, "*** IW_EV_LCP_PK2_LEN = %zu ; IW_EV_POINT_PK2_LEN = %zu\n\n",
                IW_EV_LCP_PK2_LEN, IW_EV_POINT_PK2_LEN);
    }

    /* Get range stuff */
    has_range = (iw_get_range_info(skfd, ifname, &range) >= 0);

    /* Check if the interface could support scanning. */
    if((!has_range) || (range.we_version_compiled < 14))
    {
        fprintf(stderr, "%-8.16s  Interface doesn't support scanning.\n\n",
                ifname);
        return(-1);
    }

    /* Init timeout value -> 250ms between set and first get */
    tv.tv_sec = 0;
    tv.tv_usec = 250000;

    /* Clean up set args */
    memset(&scanopt, 0, sizeof(scanopt));

    /* Parse command line arguments and extract options.
     * Note : when we have enough options, we should use the parser
     * from iwconfig... */
    while(count > 0)
    {
        /* One arg is consumed (the option name) */
        count--;

        /*
         * Check for Active Scan (scan with specific essid)
         */
        if(!strncmp(args[0], "essid", 5))
        {
            if(count < 1)
            {
                fprintf(stderr, "Too few arguments for scanning option [%s]\n",
                        args[0]);
                return(-1);
            }
            args++;
            count--;

            /* Store the ESSID in the scan options */
            scanopt.essid_len = strlen(args[0]);
            memcpy(scanopt.essid, args[0], scanopt.essid_len);
            /* Initialise BSSID as needed */
            if(scanopt.bssid.sa_family == 0)
            {
                scanopt.bssid.sa_family = ARPHRD_ETHER;
                memset(scanopt.bssid.sa_data, 0xff, ETH_ALEN);
            }
            /* Scan only this ESSID */
            scanflags |= IW_SCAN_THIS_ESSID;
        }
        else
            /* Check for last scan result (do not trigger scan) */
            if(!strncmp(args[0], "last", 4))
            {
                /* Hack */
                scanflags |= IW_SCAN_HACK;
            }
            else
            {
                fprintf(stderr, "Invalid scanning option [%s]\n", args[0]);
                return(-1);
            }

        /* Next arg */
        args++;
    }

    /* Check if we have scan options */
    if(scanflags)
    {
        wrq.u.data.pointer = (caddr_t) &scanopt;
        wrq.u.data.length = sizeof(scanopt);
        wrq.u.data.flags = scanflags;
    }
    else
    {
        wrq.u.data.pointer = NULL;
        wrq.u.data.flags = 0;
        wrq.u.data.length = 0;
    }

    /* If only 'last' was specified on command line, don't trigger a scan */
    if(scanflags == IW_SCAN_HACK)
    {
        /* Skip waiting */
        tv.tv_usec = 0;
    }
    else
    {
        /* Initiate Scanning */
        if(iw_set_ext(skfd, ifname, SIOCSIWSCAN, &wrq) < 0)
        {
            if((errno != EPERM) || (scanflags != 0))
            {
                fprintf(stderr, "%-8.16s  Interface doesn't support scanning : %s\n\n",
                        ifname, strerror(errno));
                return(-1);
            }
            /* If we don't have the permission to initiate the scan, we may
             * still have permission to read left-over results.
             * But, don't wait !!! */
#if 0
            /* Not cool, it display for non wireless interfaces... */
            fprintf(stderr, "%-8.16s  (Could not trigger scanning, just reading left-over results)\n", ifname);
#endif
            tv.tv_usec = 0;
        }
    }
    timeout -= tv.tv_usec;

    /* Forever */
    while(1)
    {
        fd_set    rfds;    /* File descriptors for select */
        int    last_fd;  /* Last fd */
        int    ret;

        /* Guess what ? We must re-generate rfds each time */
        FD_ZERO(&rfds);
        last_fd = -1;

        /* In here, add the rtnetlink fd in the list */

        /* Wait until something happens */
        ret = select(last_fd + 1, &rfds, NULL, NULL, &tv);

        /* Check if there was an error */
        if(ret < 0)
        {
            if(errno == EAGAIN || errno == EINTR)
                continue;
            fprintf(stderr, "Unhandled signal - exiting...\n");
            return(-1);
        }

        /* Check if there was a timeout */
        if(ret == 0)
        {
            unsigned char *  newbuf;

realloc:
            /* (Re)allocate the buffer - realloc(NULL, len) == malloc(len) */
            newbuf = realloc(buffer, buflen);
            if(newbuf == NULL)
            {
                if(buffer)
                    free(buffer);
                fprintf(stderr, "%s: Allocation failed\n", __FUNCTION__);
                return(-1);
            }
            buffer = newbuf;

            /* Try to read the results */
            wrq.u.data.pointer = buffer;
            wrq.u.data.flags = 0;
            wrq.u.data.length = buflen;
            if(iw_get_ext(skfd, ifname, SIOCGIWSCAN, &wrq) < 0)
            {
                /* Check if buffer was too small (WE-17 only) */
                if((errno == E2BIG) && (range.we_version_compiled > 16))
                {
                    /* Some driver may return very large scan results, either
                     * because there are many cells, or because they have many
                     * large elements in cells (like IWEVCUSTOM). Most will
                     * only need the regular sized buffer. We now use a dynamic
                     * allocation of the buffer to satisfy everybody. Of course,
                     * as we don't know in advance the size of the array, we try
                     * various increasing sizes. Jean II */

                    /* Check if the driver gave us any hints. */
                    if(wrq.u.data.length > buflen)
                        buflen = wrq.u.data.length;
                    else
                        buflen *= 2;

                    /* Try again */
                    goto realloc;
                }

                /* Check if results not available yet */
                if(errno == EAGAIN)
                {
                    /* Restart timer for only 100ms*/
                    tv.tv_sec = 0;
                    tv.tv_usec = 100000;
                    timeout -= tv.tv_usec;
                    if(timeout > 0)
                        continue;  /* Try again later */
                }

                /* Bad error */
                free(buffer);
                fprintf(stderr, "%-8.16s  Failed to read scan data : %s\n\n",
                        ifname, strerror(errno));
                return(-2);
            }
            else
                /* We have the results, go to process them */
                break;
        }

        /* In here, check if event and event type
         * if scan event, read results. All errors bad & no reset timeout */
    }

    Vector/* scan_result */ return_vals;
    if(wrq.u.data.length) {
        struct iw_event    iwe;
        struct stream_descr  stream;
        struct iwscan_state  state = { .ap_num = 1, .val_index = 0 };
        int      ret;
        inplace_make_Vector(&return_vals, 10);

#ifdef DEBUG
        /* Debugging code. In theory useless, because it's debugged ;-) */
        int  i;
        printf("Scan result %d [%02X", wrq.u.data.length, buffer[0]);
        for(i = 1; i < wrq.u.data.length; i++)
            printf(":%02X", buffer[i]);
        printf("]\n");
#endif
        //printf("%-8.16s  Scan completed :\n", ifname);
        iw_init_event_stream(&stream, (char *) buffer, wrq.u.data.length);
        do
        {
            /* Extract an event and print it */
            ret = iw_extract_event_stream(&stream, &iwe,
                    range.we_version_compiled);
            if(ret > 0)
                parse_scanning_token(&stream, &iwe, &state, &range, has_range, &return_vals);
        }
        while(ret > 0);
        //printf("\n");
    }
    else { printf("%-8.16s  No scan results\n\n", ifname); }

    char buf[40];   // overkill
    for (int i = 0; i < return_vals.size; ++i) {
        scan_result* res = return_vals.elements[i];
        iw_ether_ntop(&res->address, buf);
        printf("AP id: %s; AP name: ", buf);
        if (res->essid != NULL) { printf("%s", res->essid); }
        printf("; signal strength: ");
        if (res->signal_strength != 0) { printf("%d", res->signal_strength); }
        printf("; frequency: ");
        if (res->freq != 0) { printf("%f", res->freq); }
        printf("\n");
    }

    free(buffer);
    return(0);
}

int main() {
    int skfd = iw_sockets_open();   // Find a socket for generic queries. Don't care which type.
    print_scanning_info(skfd, "wlan0", NULL, 0);

    return 0;
}
