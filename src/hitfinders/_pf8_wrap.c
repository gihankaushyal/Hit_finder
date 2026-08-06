/* src/hitfinders/_pf8_wrap.c
 * Thin bridge between Python ctypes and CrystFEL's peakfinder8().
 *
 * Compile:
 *   cd src/hitfinders && make
 *
 * Requires CrystFEL at /data/bioxfel/software/crystfel-0.12.0/
 *
 * Signatures confirmed via objdump of libcrystfel.so v0.12.0:
 *
 *   prepare_peakfinder8(struct detgeom *dg, int fast) -> ImageDataArrays *
 *     Allocates per-panel scratch arrays sized for dg.
 *     Called internally by peakfinder8 when ida arg is NULL.
 *
 *   peakfinder8(struct image*, int max_n_peaks,
 *               float threshold, float min_snr,
 *               int min_pix_count, int max_pix_count,
 *               int local_bg_radius, int min_res, int max_res,
 *               int use_saturated, int fast,
 *               ImageDataArrays *ida)   <- 12th arg; NULL = allocate internally
 *
 * We pass ida=NULL so peakfinder8 calls prepare_peakfinder8 itself.
 * The internally-allocated ida is stored in img->ida after the call and
 * freed by image_data_arrays_free when done.
 */
#include <stdlib.h>
#include <string.h>

#include "crystfel/image.h"
#include "crystfel/detgeom.h"
#include "crystfel/peaks.h"

/* Internal symbols not in the installed public headers.
 * peakfinder8 returns ImageFeatureList* (NULL on error/no-peaks). */
extern ImageFeatureList *peakfinder8(struct image *img, int max_n_peaks,
                                     float threshold, float min_snr,
                                     int min_pix_count, int max_pix_count,
                                     int local_bg_radius, int min_res,
                                     int max_res, int use_saturated,
                                     int fast, ImageDataArrays *ida);

extern int image_feature_count(ImageFeatureList *flist);
extern struct imagefeature *image_get_feature(ImageFeatureList *flist,
                                               int idx);
extern void image_feature_list_free(ImageFeatureList *flist);

/*
 * pf8_find_peaks - Run PeakFinder8 on a single assembled detector frame.
 *
 * data            : flat float32 array of shape (h * w,), row-major
 * w, h            : frame width and height in pixels
 * threshold       : absolute minimum intensity threshold (ADU)
 * min_snr         : minimum local signal-to-noise ratio
 * min_pix_count   : minimum pixels in a connected peak
 * max_pix_count   : maximum pixels in a connected peak
 * local_bg_radius : radius for local background box (pixels)
 * min_res         : minimum distance from frame centre (0 = disabled)
 * max_res         : maximum distance from frame centre (0 = disabled)
 * use_saturated   : 1 = include pixels above max_adu, 0 = exclude
 * max_n_peaks     : capacity of out_x / out_y arrays
 * out_x           : caller-allocated float array (length max_n_peaks)
 * out_y           : caller-allocated float array (length max_n_peaks)
 * out_count       : set to actual peak count on return
 *
 * Returns 0 on success, -1 on allocation error.
 * Centroid convention: out_x[i] = column (fast-scan), out_y[i] = row (slow-scan).
 */
int pf8_find_peaks(
    const float *data, int w, int h,
    float threshold, float min_snr,
    int min_pix_count, int max_pix_count,
    int local_bg_radius, int min_res, int max_res,
    int use_saturated, int max_n_peaks,
    float *out_x, float *out_y, int *out_count
) {
    /* Build a single-panel detgeom with a minimal leaf group.
     * PF8 dereferences panel->group and detgeom->top_group. */
    struct detgeom_panel panel;
    memset(&panel, 0, sizeof(panel));
    panel.name           = "assembled";
    panel.cnx            = 0.0;
    panel.cny            = 0.0;
    panel.cnz            = 0.0;
    panel.pixel_pitch    = 1.0e-4;
    panel.adu_per_photon = 1.0;
    panel.max_adu        = 1.0e9;
    panel.fsx = 1.0;  panel.fsy = 0.0;  panel.fsz = 0.0;
    panel.ssx = 0.0;  panel.ssy = 1.0;  panel.ssz = 0.0;
    panel.w              = w;
    panel.h              = h;

    struct detgeom_panel_group leaf_group;
    memset(&leaf_group, 0, sizeof(leaf_group));
    leaf_group.name       = "assembled_group";
    leaf_group.n_children = 0;
    leaf_group.parent     = NULL;
    leaf_group.serial     = 0;
    leaf_group.children   = NULL;
    leaf_group.panel      = &panel;

    panel.group = &leaf_group;

    struct detgeom detgeom;
    memset(&detgeom, 0, sizeof(detgeom));
    detgeom.panels    = &panel;
    detgeom.n_panels  = 1;
    detgeom.top_group = &leaf_group;

    /* Copy frame data — peakfinder8 may write scratch into dp[]. */
    float *dp0 = (float *)malloc((size_t)w * (size_t)h * sizeof(float));
    if (!dp0) return -1;
    memcpy(dp0, data, (size_t)w * (size_t)h * sizeof(float));

    float **dp = (float **)malloc(sizeof(float *));
    if (!dp) { free(dp0); return -1; }
    dp[0] = dp0;

    /* NULL bad/sat arrays: no mask. */
    int   **bad = (int   **)calloc(1, sizeof(int   *));
    float **sat = (float **)calloc(1, sizeof(float *));
    if (!bad || !sat) {
        free(dp0); free(dp); free(bad); free(sat); return -1;
    }

    struct image img;
    memset(&img, 0, sizeof(img));
    img.dp      = dp;
    img.bad     = bad;
    img.sat     = sat;
    img.detgeom = &detgeom;

    /* Pass ida=NULL: peakfinder8 calls prepare_peakfinder8(detgeom, fast)
     * internally and returns the found peaks as ImageFeatureList*.
     * PF8 requires non-zero background to compute local SNR — frames with
     * uniformly zero background will return 0 peaks even for bright spots. */
    ImageFeatureList *features = peakfinder8(&img, max_n_peaks,
                                             threshold, min_snr,
                                             min_pix_count, max_pix_count,
                                             local_bg_radius, min_res, max_res,
                                             use_saturated, 0 /* fast=off */,
                                             NULL /* ida: allocate internally */);

    *out_count = 0;
    if (features != NULL) {
        int n = image_feature_count(features);
        int to_copy = (n < max_n_peaks) ? n : max_n_peaks;
        for (int i = 0; i < to_copy; i++) {
            struct imagefeature *f = image_get_feature(features, i);
            if (f) {
                out_x[*out_count] = (float)f->fs;
                out_y[*out_count] = (float)f->ss;
                (*out_count)++;
            }
        }
        image_feature_list_free(features);
    }

    if (img.ida != NULL) {
        image_data_arrays_free(img.ida);
    }

    free(dp0);
    free(dp);
    free(bad);
    free(sat);
    return 0;  /* success; *out_count holds peak count */
}
