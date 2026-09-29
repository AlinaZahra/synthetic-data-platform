/** Friendly names for locale codes, so people never have to know what "ur-PK" or "es" means. */
export const LOCALE_NAMES: Record<string, string> = {
  "en-US": "United States (English)",
  "en-GB": "United Kingdom (English)",
  "ur-PK": "Pakistan (Urdu)",
  ar: "Saudi Arabia (Arabic)",
  hi: "India (Hindi)",
  es: "Spain (Spanish)",
  fr: "France (French)",
  zh: "China (Chinese)",
};

export const localeName = (code: string, rtl?: boolean): string =>
  `${LOCALE_NAMES[code] ?? code}${rtl ? ", right-to-left" : ""}`;

export const LOCALE_HELP =
  "A country and language pack. It decides what names, phone numbers, addresses, ID numbers, dates, currency and taxes look like, so the data feels local.";
