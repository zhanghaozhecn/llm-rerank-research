-- dump_span.lua -- research-only filter for the luna mixed-window cohort
-- (2026-11-24). Packs each candidate's span / engine quality / type into the
-- comment so the headless probe can read them through the C API (whose
-- RimeCandidate struct has no start/end). Used only by the luna_mixed probe
-- schema; never loaded in production.
--
-- The comment is carried by a ShadowCandidate rather than assigned in
-- place: librime-lua's `cand.comment = x` only writes through to Phrase and
-- SimpleCandidate, so the ShadowCandidate / UniquifiedCandidate wrappers
-- emitted by the simplifier and uniquifier would silently keep an empty
-- comment (seen as blank fields for converted candidates such as 进).
local function dump_filter(input, env)
  for cand in input:iter() do
    local info = string.format("%d|%d|%.4f|%s", cand.start, cand._end,
                               cand.quality, cand.type)
    yield(ShadowCandidate(cand, cand.type, cand.text, info))
  end
end

return dump_filter
