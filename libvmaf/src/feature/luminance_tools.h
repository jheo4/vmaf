/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#ifndef LUMINANCE_TOOLS_H_
#define LUMINANCE_TOOLS_H_

typedef double (*EOTF)(double V);

typedef struct LumaRange {
    int bitdepth;
    int foot;
    int head;
} LumaRange;

LumaRange LumaRange_init(int bitdepth, const char *pix_range);

void range_foot_head(int bitdepth, const char *pix_range, int *foot, int *head);

double normalize_range(int sample, LumaRange range);

double bt1886_eotf(double V);

double get_luminance(int sample, LumaRange luma_range, EOTF eotf);

#endif