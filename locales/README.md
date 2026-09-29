Extra locale packs. Put `<code>.json` files here (same schema as `backend/sdp/locale/packs/*.json`);
they are loaded at startup when `SDP_LOCALE_DIR` points at this folder (docker-compose does this).
A pack with an unknown checksum name or missing keys fails fast at startup with a clear message.
